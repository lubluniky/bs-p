use core::pin::Pin;

pub mod ring_buffer;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct GreekOut {
    pub delta_x: f64,
    pub gamma_x: f64,
}

unsafe extern "C" {
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
    fn ffi_calculate_quotes_logit(
        x_t: *const f64,
        q_t: *const f64,
        sigma_b: *const f64,
        gamma: *const f64,
        tau: *const f64,
        k: *const f64,
        bid_p: *mut f64,
        ask_p: *mut f64,
        n: usize,
    );
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

#[allow(clippy::too_many_arguments)]
pub fn calculate_quotes_logit(
    x_t: &[f64],
    q_t: &[f64],
    sigma_b: &[f64],
    gamma: &[f64],
    tau: &[f64],
    k: &[f64],
    bid_p: &mut [f64],
    ask_p: &mut [f64],
) {
    calculate_quotes_logit_pinned(
        Pin::new(x_t),
        Pin::new(q_t),
        Pin::new(sigma_b),
        Pin::new(gamma),
        Pin::new(tau),
        Pin::new(k),
        Pin::new(bid_p),
        Pin::new(ask_p),
    );
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

#[allow(clippy::too_many_arguments)]
pub fn calculate_quotes_logit_pinned(
    x_t: Pin<&[f64]>,
    q_t: Pin<&[f64]>,
    sigma_b: Pin<&[f64]>,
    gamma: Pin<&[f64]>,
    tau: Pin<&[f64]>,
    k: Pin<&[f64]>,
    mut bid_p: Pin<&mut [f64]>,
    mut ask_p: Pin<&mut [f64]>,
) {
    let x_ref = x_t.get_ref();
    let q_ref = q_t.get_ref();
    let sigma_ref = sigma_b.get_ref();
    let gamma_ref = gamma.get_ref();
    let tau_ref = tau.get_ref();
    let k_ref = k.get_ref();
    let bid_ref = bid_p.as_mut().get_mut();
    let ask_ref = ask_p.as_mut().get_mut();

    let n = x_ref.len();
    assert_eq!(q_ref.len(), n, "calculate_quotes_logit: q_t length mismatch");
    assert_eq!(
        sigma_ref.len(),
        n,
        "calculate_quotes_logit: sigma_b length mismatch"
    );
    assert_eq!(
        gamma_ref.len(),
        n,
        "calculate_quotes_logit: gamma length mismatch"
    );
    assert_eq!(tau_ref.len(), n, "calculate_quotes_logit: tau length mismatch");
    assert_eq!(k_ref.len(), n, "calculate_quotes_logit: k length mismatch");
    assert_eq!(
        bid_ref.len(),
        n,
        "calculate_quotes_logit: bid_p length mismatch"
    );
    assert_eq!(
        ask_ref.len(),
        n,
        "calculate_quotes_logit: ask_p length mismatch"
    );

    unsafe {
        ffi_calculate_quotes_logit(
            x_ref.as_ptr(),
            q_ref.as_ptr(),
            sigma_ref.as_ptr(),
            gamma_ref.as_ptr(),
            tau_ref.as_ptr(),
            k_ref.as_ptr(),
            bid_ref.as_mut_ptr(),
            ask_ref.as_mut_ptr(),
            n,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{align_of, size_of};

    #[test]
    fn greek_layout_is_compact() {
        assert_eq!(size_of::<GreekOut>(), 16);
        assert_eq!(align_of::<GreekOut>(), 8);
    }
}
