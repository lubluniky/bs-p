#ifndef PM_KERNEL_ANALYTICS_H
#define PM_KERNEL_ANALYTICS_H

#include <stddef.h>

#include "kernel.h"

#ifdef __cplusplus
extern "C" {
#endif

void implied_belief_volatility_batch(
    const double *bid_p,
    const double *ask_p,
    const double *q_t,
    const double *gamma,
    const double *tau,
    const double *k,
    double *out_sigma_b,
    size_t n
);

void simulate_shock_logit_batch(
    const double *x_t,
    const double *q_t,
    const double *sigma_b,
    const double *gamma,
    const double *tau,
    const double *k,
    const double *shock_p,
    double *out_r_x,
    double *out_bid_p,
    double *out_ask_p,
    greek_out_t *out_greeks,
    double *out_pnl_shift,
    size_t n
);

void adaptive_kelly_clip_batch(
    const double *belief_p,
    const double *market_p,
    const double *q_t,
    const double *gamma,
    const double *risk_limit,
    const double *max_clip,
    double *out_maker_clip,
    double *out_taker_clip,
    size_t n
);

void order_book_microstructure_batch(
    const double *bid_p,
    const double *ask_p,
    const double *bid_vol,
    const double *ask_vol,
    double *out_obi,
    double *out_vwm_p,
    double *out_vwm_x,
    double *out_pressure,
    size_t n
);

void aggregate_portfolio_greeks(
    const double *positions,
    const double *delta_x,
    const double *gamma_x,
    const double *weights,
    const double *corr_matrix,
    size_t n,
    double *out_net_delta,
    double *out_net_gamma
);

#ifdef __cplusplus
}
#endif

#endif
