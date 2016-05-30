
#[macro_export]
macro_rules! try_opt {
    ($e: expr) => (match $e { Some(e) => e, None => return None })
}

pub fn min_index<T: PartialOrd>(arr: &[T]) -> usize {
    let mut min_idx = 0;
    for i in 1..4 {
        if arr[i] < arr[min_idx] {
            min_idx = i;
        }
    }
    min_idx
}

pub fn max_index<T: PartialOrd>(arr: &[T]) -> usize {
    let mut max_idx = 0;
    for i in 1..arr.len() {
        if arr[i] > arr[max_idx] {
            max_idx = i;
        }
    }
    max_idx
}


