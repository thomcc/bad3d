use super::watched_file::WatchedFile;
use failure::ResultExt;
use glium::backend::Facade;
// use handy::{Handle, HandleMap};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
#[derive(Debug, Clone)]
pub struct ShaderConf {
    pub repo_relative_path: &'static str,
    pub override_env_var: Option<&'static str>,
    pub preamble_file: Option<&'static str>,
    pub fs_suffix: &'static str,
    pub vs_suffix: &'static str,
    pub fs_define: Option<&'static str>,
    pub vs_define: Option<&'static str>,
}

struct ShaderInfo {
    name: String,
    vs_path: PathBuf,
    fs_path: PathBuf,

    program: glium::Program,
    vs_src: String,
    fs_src: String,
    fs_watcher: WatchedFile,
    vs_watcher: WatchedFile,
}

pub struct ShaderLib {
    shader_dir: PathBuf,
    items: HashMap<String, ShaderInfo>,
    // watchers: HandleMap<(WatchedFile, Handle)>,
    preamble: Option<(WatchedFile, String)>,
    conf: ShaderConf,
}

fn add_path(p: impl AsRef<Path>, child: impl AsRef<Path>) -> PathBuf {
    let mut p: PathBuf = p.as_ref().into();
    p.push(child);
    p
}

impl ShaderLib {
    pub fn new(conf: ShaderConf) -> failure::Fallible<Self> {
        // let w = notify::watcher(std::time::Duration::from_secs(1));
        let shader_dir = conf
            .override_env_var
            .and_then(|v| std::env::var_os(v).map(PathBuf::from))
            .or_else(|| {
                std::env::var_os("CARGO_MANIFEST_DIR").and_then(|s| path_with_child_dir(s, conf.repo_relative_path))
            })
            .or_else(|| Self::find_path(&conf));
        let shader_dir = if let Some(p) = shader_dir {
            p
        } else {
            failure::bail!("Failed to locate shader path");
        };
        let preamble = conf
            .preamble_file
            .map(|p| WatchedFile::new_with_text(add_path(&shader_dir, p)))
            .transpose()?;

        Ok(Self {
            shader_dir,
            preamble,
            conf,
            items: Default::default(),
        })
    }

    pub fn get_shader(&self, s: &str) -> &glium::Program {
        &self.items[s].program
        // self.items.iter().find(|si| &si.name == s).map(|si| &si.program)
    }

    pub fn get_shader_mut(&mut self, s: &str) -> &mut glium::Program {
        &mut self.items.get_mut(s).unwrap().program
    }

    pub fn try_get_shader(&self, s: &str) -> Option<&glium::Program> {
        self.items.get(s).map(|s| &s.program)
    }

    pub fn try_get_shader_mut(&mut self, s: &str) -> Option<&mut glium::Program> {
        self.items.get_mut(s).map(|s| &mut s.program)
    }

    pub fn maybe_reload_shaders<F: Facade>(&mut self, f: &F) -> failure::Fallible<()> {
        let mut preamble_changed = false;
        if let Some((watcher, src)) = &mut self.preamble {
            if let Some(t) = watcher.maybe_reload_text()? {
                log::info!("Preamble changed, recompiling all programs");
                *src = t;
                preamble_changed = true;
            }
        }
        let preamble = self.preamble.as_ref().map(|p| p.1.as_str());
        let mut all_good = true;
        for s in self.items.values_mut() {
            let mut changed = preamble_changed;
            let mut stage = |src: &mut String, watcher: &mut WatchedFile, path: &Path| match watcher.maybe_reload_text()
            {
                Ok(Some(new_src)) => {
                    *src = new_src;
                    changed = true;
                }
                Err(e) => {
                    log::error!("Failed to read file {:?}: {}", path, e);
                    all_good = false;
                }
                _ => {}
            };
            stage(&mut s.fs_src, &mut s.fs_watcher, &s.fs_path);
            stage(&mut s.vs_src, &mut s.vs_watcher, &s.vs_path);
            if changed {
                log::debug!("Change in {} detected, recompiling", s.name);
                match compile(f, preamble, &s.fs_src, &s.vs_src) {
                    Ok(program) => {
                        log::debug!("Recompiled {}", s.name);
                        s.program = program;
                    }
                    Err(e) => {
                        log::error!("Error in {}: {}", s.name, e);
                        all_good = false;
                    }
                }
            }
        }
        if !all_good {
            eprintln!("Some shaders failed to compile D:");
        }
        Ok(())
    }

    pub fn add<F: Facade>(&mut self, f: &F, shaders: &[&str]) -> failure::Fallible<()> {
        for &name in shaders {
            if self.items.contains_key(name) {
                log::info!("Already loaded {}", name);
                continue;
            }

            let name = name.to_string();
            let fp = add_path(&self.shader_dir, name.clone() + self.conf.fs_suffix);
            let vp = add_path(&self.shader_dir, name.clone() + self.conf.vs_suffix);

            log::info!("Loading shader {}", name);
            log::trace!("  frag from {:?}", fp);
            let (fs_watcher, fs_src) = WatchedFile::new_with_text(fp.clone())?;
            log::trace!("  vert from {:?}", vp);
            let (vs_watcher, vs_src) = WatchedFile::new_with_text(vp.clone())?;
            let preamble = self.preamble.as_ref().map(|p| p.1.as_str());
            let program = match compile(f, preamble, &fs_src, &vs_src) {
                Ok(v) => v,
                Err(e) => {
                    log::error!("Error compiling {}: {}", name, e);
                    return Err(e.into());
                }
            };
            let shader_info = ShaderInfo {
                name: name.clone(),
                program,
                fs_src,
                vs_src,
                fs_path: fp,
                vs_path: vp,
                fs_watcher: fs_watcher.with_interval(std::time::Duration::from_millis(250)),
                vs_watcher: vs_watcher.with_interval(std::time::Duration::from_millis(250)),
            };
            self.items.insert(name, shader_info);
        }
        Ok(())
    }

    fn find_path(conf: &ShaderConf) -> Option<PathBuf> {
        let to_search: &[(&str, fn() -> std::io::Result<PathBuf>, bool)] = &[
            ("current_dir", std::env::current_dir, false),
            ("current_exe", std::env::current_exe, true),
        ];
        // if given blah/resources/shaders as the repo-relative path, look for
        // `resources/shaders` and `shaders` too.
        let parts = conf.repo_relative_path.split("/").collect::<Vec<_>>();
        let mut subdirs_to_search = Vec::with_capacity(parts.len());
        subdirs_to_search.push(conf.repo_relative_path.to_owned());
        for i in 0..(parts.len() - 1) {
            let rest = &parts[i..];
            if !rest.is_empty() {
                let path = rest.join("/");
                subdirs_to_search.push(path);
            }
        }

        for (what, how, search_parents) in to_search {
            let mut d = match how() {
                Ok(d) => d,
                Err(e) => {
                    log::warn!("Failed to run {}: {:?}", what, e);
                    continue;
                }
            };
            log::debug!("Searching from {}: {:?}", what, d);
            log::debug!("  will check `$path/shaders`, `path/{}`.", conf.repo_relative_path);
            log::debug!("  check parents too? (stops at Cargo.toml) {}", search_parents);
            loop {
                log::trace!("  Looking for shaders in {:?}", d);
                for subdir in &subdirs_to_search {
                    log::trace!("    Checking subdir {}", subdir);
                    if let Some(d) = path_with_child_dir(&d, subdir) {
                        log::trace!("    Found: {:?}", d);
                        return Some(d);
                    }
                }
                // The main reason for searching parents is to find things if
                // we're in `target/blah/release` or something.
                if *search_parents && path_has_child(&d, "Cargo.toml") {
                    log::trace!("Stopping search of parents from {} at {:?} (See Cargo.toml)", what, d);
                    break;
                }
                if !search_parents || !d.pop() {
                    break;
                }
                log::trace!("  Not found, checking parent...");
            }
        }
        log::error!("Couldn't find shader path");
        None
    }
}
fn compile<F: Facade>(
    f: &F,
    preamble: Option<&str>,
    frag: &str,
    vert: &str,
) -> Result<glium::Program, glium::ProgramCreationError> {
    if let Some(preamble) = preamble {
        let frag_src = format!("{}\n#line 1\n{}", preamble, frag);
        let vert_src = format!("{}\n#line 1\n{}", preamble, vert);
        glium::Program::from_source(f, &vert_src, &frag_src, None)
    } else {
        glium::Program::from_source(f, &vert, &frag, None)
    }
}

fn path_has_child(p: impl AsRef<Path>, child: impl AsRef<Path>) -> bool {
    let mut p: PathBuf = p.as_ref().into();
    p.push(child);
    p.exists()
}

fn path_with_child_dir(p: impl AsRef<Path>, child: impl AsRef<Path>) -> Option<PathBuf> {
    let mut p: PathBuf = p.as_ref().into();
    p.push(child);
    if p.is_dir() && p.exists() {
        Some(p)
    } else {
        None
    }
}

impl<'a> std::ops::Index<&'a str> for ShaderLib {
    type Output = glium::Program;
    fn index(&self, s: &'a str) -> &glium::Program {
        self.get_shader(s)
    }
}
