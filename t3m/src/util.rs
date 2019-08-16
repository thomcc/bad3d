#[derive(Copy, Clone)]
#[repr(C, align(16))]
pub struct Align16<T: Copy>(pub T);

#[derive(Copy, Clone)]
pub union ConstTransmuter<From: Copy, To: Copy> {
    pub from: From,
    pub to: To,
}
