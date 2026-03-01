use core::ptr;

use crate::GreekOut;

unsafe extern "C" {
    #[link_name = "implied_belief_volatility_batch"]
    fn ffi_implied_belief_volatility_batch(
        bid_p: *const f64,
        ask_p: *const f64,
        q_t: *const f64,
        gamma: *const f64,
        tau: *const f64,
        k: *const f64,
        out_sigma_b: *mut f64,
        n: usize,
    );

    #[link_name = "simulate_shock_logit_batch"]
    fn ffi_simulate_shock_logit_batch(
        x_t: *const f64,
        q_t: *const f64,
        sigma_b: *const f64,
        gamma: *const f64,
        tau: *const f64,
        k: *const f64,
        shock_p: *const f64,
        out_r_x: *mut f64,
        out_bid_p: *mut f64,
        out_ask_p: *mut f64,
        out_greeks: *mut GreekOut,
        out_pnl_shift: *mut f64,
        n: usize,
    );

    #[link_name = "adaptive_kelly_clip_batch"]
    fn ffi_adaptive_kelly_clip_batch(
        belief_p: *const f64,
        market_p: *const f64,
        q_t: *const f64,
        gamma: *const f64,
        risk_limit: *const f64,
        max_clip: *const f64,
        out_maker_clip: *mut f64,
        out_taker_clip: *mut f64,
        n: usize,
    );

    #[link_name = "order_book_microstructure_batch"]
    fn ffi_order_book_microstructure_batch(
        bid_p: *const f64,
        ask_p: *const f64,
        bid_vol: *const f64,
        ask_vol: *const f64,
        out_obi: *mut f64,
        out_vwm_p: *mut f64,
        out_vwm_x: *mut f64,
        out_pressure: *mut f64,
        n: usize,
    );

    #[link_name = "aggregate_portfolio_greeks"]
    fn ffi_aggregate_portfolio_greeks(
        positions: *const f64,
        delta_x: *const f64,
        gamma_x: *const f64,
        weights: *const f64,
        corr_matrix: *const f64,
        n: usize,
        out_net_delta: *mut f64,
        out_net_gamma: *mut f64,
    );
}

fn assert_len(label: &str, actual: usize, expected: usize) {
    assert_eq!(
        actual, expected,
        "{label}: length mismatch (expected {expected}, got {actual})"
    );
}

#[allow(clippy::too_many_arguments)]
pub fn implied_belief_volatility_batch(
    bid_p: &[f64],
    ask_p: &[f64],
    q_t: &[f64],
    gamma: &[f64],
    tau: &[f64],
    k: &[f64],
    out_sigma_b: &mut [f64],
) {
    let n = bid_p.len();
    assert_len("implied_belief_volatility_batch: ask_p", ask_p.len(), n);
    assert_len("implied_belief_volatility_batch: q_t", q_t.len(), n);
    assert_len("implied_belief_volatility_batch: gamma", gamma.len(), n);
    assert_len("implied_belief_volatility_batch: tau", tau.len(), n);
    assert_len("implied_belief_volatility_batch: k", k.len(), n);
    assert_len(
        "implied_belief_volatility_batch: out_sigma_b",
        out_sigma_b.len(),
        n,
    );

    unsafe {
        ffi_implied_belief_volatility_batch(
            bid_p.as_ptr(),
            ask_p.as_ptr(),
            q_t.as_ptr(),
            gamma.as_ptr(),
            tau.as_ptr(),
            k.as_ptr(),
            out_sigma_b.as_mut_ptr(),
            n,
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub fn simulate_shock_logit_batch(
    x_t: &[f64],
    q_t: &[f64],
    sigma_b: &[f64],
    gamma: &[f64],
    tau: &[f64],
    k: &[f64],
    shock_p: &[f64],
    out_r_x: &mut [f64],
    out_bid_p: &mut [f64],
    out_ask_p: &mut [f64],
    out_greeks: &mut [GreekOut],
    out_pnl_shift: &mut [f64],
) {
    let n = x_t.len();
    assert_len("simulate_shock_logit_batch: q_t", q_t.len(), n);
    assert_len("simulate_shock_logit_batch: sigma_b", sigma_b.len(), n);
    assert_len("simulate_shock_logit_batch: gamma", gamma.len(), n);
    assert_len("simulate_shock_logit_batch: tau", tau.len(), n);
    assert_len("simulate_shock_logit_batch: k", k.len(), n);
    assert_len("simulate_shock_logit_batch: shock_p", shock_p.len(), n);
    assert_len("simulate_shock_logit_batch: out_r_x", out_r_x.len(), n);
    assert_len("simulate_shock_logit_batch: out_bid_p", out_bid_p.len(), n);
    assert_len("simulate_shock_logit_batch: out_ask_p", out_ask_p.len(), n);
    assert_len(
        "simulate_shock_logit_batch: out_greeks",
        out_greeks.len(),
        n,
    );
    assert_len(
        "simulate_shock_logit_batch: out_pnl_shift",
        out_pnl_shift.len(),
        n,
    );

    unsafe {
        ffi_simulate_shock_logit_batch(
            x_t.as_ptr(),
            q_t.as_ptr(),
            sigma_b.as_ptr(),
            gamma.as_ptr(),
            tau.as_ptr(),
            k.as_ptr(),
            shock_p.as_ptr(),
            out_r_x.as_mut_ptr(),
            out_bid_p.as_mut_ptr(),
            out_ask_p.as_mut_ptr(),
            out_greeks.as_mut_ptr(),
            out_pnl_shift.as_mut_ptr(),
            n,
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub fn adaptive_kelly_clip_batch(
    belief_p: &[f64],
    market_p: &[f64],
    q_t: &[f64],
    gamma: &[f64],
    risk_limit: &[f64],
    max_clip: &[f64],
    out_maker_clip: &mut [f64],
    out_taker_clip: &mut [f64],
) {
    let n = belief_p.len();
    assert_len("adaptive_kelly_clip_batch: market_p", market_p.len(), n);
    assert_len("adaptive_kelly_clip_batch: q_t", q_t.len(), n);
    assert_len("adaptive_kelly_clip_batch: gamma", gamma.len(), n);
    assert_len("adaptive_kelly_clip_batch: risk_limit", risk_limit.len(), n);
    assert_len("adaptive_kelly_clip_batch: max_clip", max_clip.len(), n);
    assert_len(
        "adaptive_kelly_clip_batch: out_maker_clip",
        out_maker_clip.len(),
        n,
    );
    assert_len(
        "adaptive_kelly_clip_batch: out_taker_clip",
        out_taker_clip.len(),
        n,
    );

    unsafe {
        ffi_adaptive_kelly_clip_batch(
            belief_p.as_ptr(),
            market_p.as_ptr(),
            q_t.as_ptr(),
            gamma.as_ptr(),
            risk_limit.as_ptr(),
            max_clip.as_ptr(),
            out_maker_clip.as_mut_ptr(),
            out_taker_clip.as_mut_ptr(),
            n,
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub fn order_book_microstructure_batch(
    bid_p: &[f64],
    ask_p: &[f64],
    bid_vol: &[f64],
    ask_vol: &[f64],
    out_obi: &mut [f64],
    out_vwm_p: &mut [f64],
    out_vwm_x: &mut [f64],
    out_pressure: &mut [f64],
) {
    let n = bid_p.len();
    assert_len("order_book_microstructure_batch: ask_p", ask_p.len(), n);
    assert_len("order_book_microstructure_batch: bid_vol", bid_vol.len(), n);
    assert_len("order_book_microstructure_batch: ask_vol", ask_vol.len(), n);
    assert_len("order_book_microstructure_batch: out_obi", out_obi.len(), n);
    assert_len(
        "order_book_microstructure_batch: out_vwm_p",
        out_vwm_p.len(),
        n,
    );
    assert_len(
        "order_book_microstructure_batch: out_vwm_x",
        out_vwm_x.len(),
        n,
    );
    assert_len(
        "order_book_microstructure_batch: out_pressure",
        out_pressure.len(),
        n,
    );

    unsafe {
        ffi_order_book_microstructure_batch(
            bid_p.as_ptr(),
            ask_p.as_ptr(),
            bid_vol.as_ptr(),
            ask_vol.as_ptr(),
            out_obi.as_mut_ptr(),
            out_vwm_p.as_mut_ptr(),
            out_vwm_x.as_mut_ptr(),
            out_pressure.as_mut_ptr(),
            n,
        );
    }
}

pub fn aggregate_portfolio_greeks(
    positions: &[f64],
    delta_x: &[f64],
    gamma_x: &[f64],
    weights: Option<&[f64]>,
    corr_matrix: Option<&[f64]>,
) -> (f64, f64) {
    let n = positions.len();
    assert_len("aggregate_portfolio_greeks: delta_x", delta_x.len(), n);
    assert_len("aggregate_portfolio_greeks: gamma_x", gamma_x.len(), n);

    if let Some(w) = weights {
        assert_len("aggregate_portfolio_greeks: weights", w.len(), n);
    }

    if let Some(corr) = corr_matrix {
        let expected = n
            .checked_mul(n)
            .expect("aggregate_portfolio_greeks: n*n overflow");
        assert_len(
            "aggregate_portfolio_greeks: corr_matrix",
            corr.len(),
            expected,
        );
    }

    let mut net_delta = 0.0;
    let mut net_gamma = 0.0;

    unsafe {
        ffi_aggregate_portfolio_greeks(
            positions.as_ptr(),
            delta_x.as_ptr(),
            gamma_x.as_ptr(),
            weights.map_or(ptr::null(), <[f64]>::as_ptr),
            corr_matrix.map_or(ptr::null(), <[f64]>::as_ptr),
            n,
            &mut net_delta,
            &mut net_gamma,
        );
    }

    (net_delta, net_gamma)
}
