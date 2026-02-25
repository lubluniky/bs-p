use core::pin::Pin;

#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, Default)]
pub struct MarketState {
    pub x_t: f64,
    pub q_t: f64,
    pub sigma_b: f64,
    pub gamma: f64,
    pub tau: f64,
    pub k: f64,
    pub reserved0: f64,
    pub reserved1: f64,
}

impl MarketState {
    #[inline]
    pub const fn new(x_t: f64, q_t: f64, sigma_b: f64, gamma: f64, tau: f64, k: f64) -> Self {
        Self {
            x_t,
            q_t,
            sigma_b,
            gamma,
            tau,
            k,
            reserved0: 0.0,
            reserved1: 0.0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct QuoteOut {
    pub bid_p: f64,
    pub ask_p: f64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct GreekOut {
    pub delta_x: f64,
    pub gamma_x: f64,
}

extern "C" {
    #[link_name = "kernel_sigmoid"]
    fn ffi_kernel_sigmoid(x: f64) -> f64;
    #[link_name = "kernel_logit"]
    fn ffi_kernel_logit(p: f64) -> f64;
    #[link_name = "kernel_sigmoid_batch"]
    fn ffi_kernel_sigmoid_batch(x: *const f64, out_p: *mut f64, n: usize);
    #[link_name = "kernel_logit_batch"]
    fn ffi_kernel_logit_batch(p: *const f64, out_x: *mut f64, n: usize);
    #[link_name = "kernel_greeks_from_logit"]
    fn ffi_kernel_greeks_from_logit(x: f64, delta_x: *mut f64, gamma_x: *mut f64);
    #[link_name = "kernel_greeks_batch"]
    fn ffi_kernel_greeks_batch(x: *const f64, out: *mut GreekOut, n: usize);
    #[link_name = "calculate_quotes_logit"]
    fn ffi_calculate_quotes_logit(states: *const MarketState, quotes: *mut QuoteOut, n: usize);
}

#[inline]
pub fn sigmoid(x: f64) -> f64 {
    unsafe { ffi_kernel_sigmoid(x) }
}

#[inline]
pub fn logit(p: f64) -> f64 {
    unsafe { ffi_kernel_logit(p) }
}

#[inline]
pub fn greeks_from_logit(x: f64) -> GreekOut {
    let mut out = GreekOut::default();
    unsafe {
        ffi_kernel_greeks_from_logit(x, &mut out.delta_x, &mut out.gamma_x);
    }
    out
}

pub fn sigmoid_batch(x: &[f64], out_p: &mut [f64]) {
    sigmoid_batch_pinned(Pin::new(x), Pin::new(out_p));
}

pub fn logit_batch(p: &[f64], out_x: &mut [f64]) {
    logit_batch_pinned(Pin::new(p), Pin::new(out_x));
}

pub fn greeks_batch(x: &[f64], out: &mut [GreekOut]) {
    greeks_batch_pinned(Pin::new(x), Pin::new(out));
}

pub fn calculate_quotes_logit(states: &[MarketState], quotes: &mut [QuoteOut]) {
    calculate_quotes_logit_pinned(Pin::new(states), Pin::new(quotes));
}

pub fn sigmoid_batch_pinned(x: Pin<&[f64]>, mut out_p: Pin<&mut [f64]>) {
    let x_ref = x.get_ref();
    let out_ref = out_p.as_mut().get_mut();
    assert_eq!(
        x_ref.len(),
        out_ref.len(),
        "sigmoid_batch: input and output lengths must match"
    );

    unsafe {
        ffi_kernel_sigmoid_batch(x_ref.as_ptr(), out_ref.as_mut_ptr(), x_ref.len());
    }
}

pub fn logit_batch_pinned(p: Pin<&[f64]>, mut out_x: Pin<&mut [f64]>) {
    let p_ref = p.get_ref();
    let out_ref = out_x.as_mut().get_mut();
    assert_eq!(
        p_ref.len(),
        out_ref.len(),
        "logit_batch: input and output lengths must match"
    );

    unsafe {
        ffi_kernel_logit_batch(p_ref.as_ptr(), out_ref.as_mut_ptr(), p_ref.len());
    }
}

pub fn greeks_batch_pinned(x: Pin<&[f64]>, mut out: Pin<&mut [GreekOut]>) {
    let x_ref = x.get_ref();
    let out_ref = out.as_mut().get_mut();
    assert_eq!(
        x_ref.len(),
        out_ref.len(),
        "greeks_batch: input and output lengths must match"
    );

    unsafe {
        ffi_kernel_greeks_batch(x_ref.as_ptr(), out_ref.as_mut_ptr(), x_ref.len());
    }
}

pub fn calculate_quotes_logit_pinned(
    states: Pin<&[MarketState]>,
    mut quotes: Pin<&mut [QuoteOut]>,
) {
    let states_ref = states.get_ref();
    let quotes_ref = quotes.as_mut().get_mut();
    assert_eq!(
        states_ref.len(),
        quotes_ref.len(),
        "calculate_quotes_logit: input and output lengths must match"
    );

    unsafe {
        ffi_calculate_quotes_logit(states_ref.as_ptr(), quotes_ref.as_mut_ptr(), states_ref.len());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{align_of, size_of};

    #[test]
    fn market_state_layout_matches_c() {
        assert_eq!(size_of::<MarketState>(), 64);
        assert_eq!(align_of::<MarketState>(), 64);
    }

    #[test]
    fn quote_layout_is_compact() {
        assert_eq!(size_of::<QuoteOut>(), 16);
        assert_eq!(align_of::<QuoteOut>(), 8);
    }

    #[test]
    fn greek_layout_is_compact() {
        assert_eq!(size_of::<GreekOut>(), 16);
        assert_eq!(align_of::<GreekOut>(), 8);
    }
}
