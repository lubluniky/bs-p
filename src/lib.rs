use core::pin::Pin;

pub mod analytics;
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
    assert_eq!(
        q_ref.len(),
        n,
        "calculate_quotes_logit: q_t length mismatch"
    );
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
    assert_eq!(
        tau_ref.len(),
        n,
        "calculate_quotes_logit: tau length mismatch"
    );
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

    unsafe extern "C" {
        #[link_name = "pm_internal_calculate_quotes_logit_portable"]
        fn ffi_internal_calculate_quotes_logit_portable(
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

    fn assert_close(actual: f64, expected: f64, tol: f64) {
        let diff = (actual - expected).abs();
        assert!(
            diff <= tol,
            "expected {expected:.15}, got {actual:.15}, diff {diff:.15} > {tol:.15}"
        );
    }

    fn has_avx512f() -> bool {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            std::arch::is_x86_feature_detected!("avx512f")
        }

        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            false
        }
    }

    fn run_quote_portable(
        x_t: &[f64],
        q_t: &[f64],
        sigma_b: &[f64],
        gamma: &[f64],
        tau: &[f64],
        k: &[f64],
    ) -> (Vec<f64>, Vec<f64>) {
        let n = x_t.len();
        let mut bid_p = vec![0.0; n];
        let mut ask_p = vec![0.0; n];

        unsafe {
            ffi_internal_calculate_quotes_logit_portable(
                x_t.as_ptr(),
                q_t.as_ptr(),
                sigma_b.as_ptr(),
                gamma.as_ptr(),
                tau.as_ptr(),
                k.as_ptr(),
                bid_p.as_mut_ptr(),
                ask_p.as_mut_ptr(),
                n,
            );
        }

        (bid_p, ask_p)
    }

    #[test]
    fn greek_layout_is_compact() {
        assert_eq!(size_of::<GreekOut>(), 16);
        assert_eq!(align_of::<GreekOut>(), 8);
    }

    #[test]
    fn logit_sigmoid_roundtrip_is_stable() {
        for x in [-20.0, -10.0, -2.0, 0.0, 2.0, 10.0, 20.0] {
            let p = sigmoid(x);
            let roundtrip = logit(p);
            assert_close(roundtrip, x, 1e-7);
        }
    }

    #[test]
    fn sigmoid_batch_matches_scalar() {
        let x = [-20.0, -3.0, -0.5, 0.0, 0.5, 4.0, 12.0, 20.0, -15.0];
        let mut out = vec![0.0; x.len()];
        sigmoid_batch(&x, &mut out);

        for (actual, input) in out.iter().zip(x) {
            assert_close(*actual, sigmoid(input), 1e-15);
        }
    }

    #[test]
    fn greeks_batch_matches_scalar_reference() {
        let x = [-12.0, -1.5, 0.0, 1.5, 12.0, 20.0, -20.0];
        let mut out = vec![GreekOut::default(); x.len()];
        greeks_batch(&x, &mut out);

        for (actual, input) in out.iter().zip(x) {
            let expected = greeks_from_logit(input);
            assert_close(actual.delta_x, expected.delta_x, 1e-15);
            assert_close(actual.gamma_x, expected.gamma_x, 1e-15);
        }
    }

    #[test]
    fn calculate_quotes_dispatch_matches_portable_reference() {
        if !has_avx512f() {
            return;
        }

        for len in [3usize, 8, 17] {
            let x_t: Vec<_> = (0..len)
                .map(|i| -2.5 + (i as f64) * 0.4 + if i % 4 == 0 { 6.0 } else { 0.0 })
                .collect();
            let q_t: Vec<_> = (0..len).map(|i| (i as f64) - 3.0).collect();
            let sigma_b: Vec<_> = (0..len).map(|i| 0.1 + (i as f64) * 0.02).collect();
            let gamma: Vec<_> = (0..len)
                .map(|i| {
                    if i % 5 == 0 {
                        -0.05
                    } else {
                        0.03 + (i as f64) * 0.01
                    }
                })
                .collect();
            let tau: Vec<_> = (0..len)
                .map(|i| {
                    if i % 6 == 0 {
                        -0.1
                    } else {
                        0.2 + (i as f64) * 0.03
                    }
                })
                .collect();
            let k: Vec<_> = (0..len)
                .map(|i| {
                    if i % 7 == 0 {
                        0.0
                    } else {
                        1.0 + (i as f64) * 0.1
                    }
                })
                .collect();

            let mut bid_p = vec![0.0; len];
            let mut ask_p = vec![0.0; len];
            calculate_quotes_logit(
                &x_t, &q_t, &sigma_b, &gamma, &tau, &k, &mut bid_p, &mut ask_p,
            );

            let (expected_bid, expected_ask) =
                run_quote_portable(&x_t, &q_t, &sigma_b, &gamma, &tau, &k);

            for i in 0..len {
                assert_close(bid_p[i], expected_bid[i], 1e-12);
                assert_close(ask_p[i], expected_ask[i], 1e-12);
            }
        }
    }
}
