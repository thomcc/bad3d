//! Note: Taken from some other code I had lying around...
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};
#[derive(Debug)]
pub struct WatchedFile {
    path: PathBuf,
    read_at: SystemTime,
    last_check: SystemTime,
    first_time: bool,
    check_interval: Option<Duration>,
}

impl WatchedFile {
    pub fn new<P>(p: P) -> Self
    where
        P: Into<PathBuf>,
    {
        Self {
            path: p.into(),
            read_at: SystemTime::UNIX_EPOCH,
            last_check: SystemTime::UNIX_EPOCH,
            first_time: true,
            check_interval: Some(Duration::from_secs(2)),
        }
    }

    pub fn new_with_text<P>(p: P) -> Result<(Self, String), std::io::Error>
    where
        P: Into<PathBuf>,
    {
        Self::new_with_contents(p, |f| std::fs::read_to_string(f))
    }

    pub fn new_with_bytes<P>(p: P) -> Result<(Self, Vec<u8>), std::io::Error>
    where
        P: Into<PathBuf>,
    {
        Self::new_with_contents(p, |f| std::fs::read(f))
    }

    pub fn new_with_contents<F, T, E, P>(p: P, cb: F) -> Result<(Self, T), E>
    where
        F: FnOnce(&Path) -> Result<T, E>,
        P: Into<PathBuf>,
    {
        let mut r = Self::new(p);
        let v = r.force_load_with(cb)?;
        Ok((r, v))
    }

    pub fn with_interval(mut self, d: impl Into<Option<Duration>>) -> Self {
        self.check_interval = d.into();
        self
    }
    pub fn set_interval(&mut self, d: impl Into<Option<Duration>>) {
        self.check_interval = d.into();
    }
    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn last_mod_time(&self) -> Option<SystemTime> {
        std::fs::metadata(&self.path)
            .and_then(|m| m.modified())
            .map_err(|e| {
                log::warn!("Failed to read modification time of {:?}: {}", self.path, e);
            })
            .ok()
    }

    pub fn need_reload(&self) -> bool {
        self.last_mod_time().map_or(false, |mod_at| mod_at > self.read_at)
    }

    pub fn maybe_reload_with<F, T, E>(&mut self, cb: F) -> Result<Option<T>, E>
    where
        F: FnOnce(&Path) -> Result<T, E>,
    {
        if self.first_time {
            self.first_time = false;
            self.last_check = SystemTime::now();
            self.read_at = SystemTime::now();
            return cb(&self.path).map(Some);
        }
        if let Some(check_interval) = self.check_interval {
            if self.last_check.elapsed().ok().map_or(false, |d| d < check_interval) {
                return Ok(None);
            }
        }
        match self.last_mod_time() {
            Some(t) => {
                self.last_check = SystemTime::now();
                if t > self.read_at {
                    self.read_at = SystemTime::now();
                    cb(&self.path).map(Some)
                } else {
                    Ok(None)
                }
            }
            _ => {
                self.last_check = SystemTime::now();
                Ok(None)
            }
        }
    }

    pub fn force_load_with<F, T, E>(&mut self, cb: F) -> Result<T, E>
    where
        F: FnOnce(&Path) -> Result<T, E>,
    {
        self.last_check = SystemTime::now();
        self.read_at = SystemTime::now();
        self.first_time = false;
        cb(&self.path)
    }

    pub fn maybe_reload_text(&mut self) -> Result<Option<String>, std::io::Error> {
        self.maybe_reload_with(|p| std::fs::read_to_string(p))
    }

    pub fn maybe_reload_bytes(&mut self) -> Result<Option<Vec<u8>>, std::io::Error> {
        self.maybe_reload_with(|p| std::fs::read(p))
    }

    pub fn force_load_text(&mut self) -> Result<String, std::io::Error> {
        self.force_load_with(|p| std::fs::read_to_string(p))
    }

    pub fn force_load_bytes(&mut self) -> Result<Vec<u8>, std::io::Error> {
        self.force_load_with(|p| std::fs::read(p))
    }
}
