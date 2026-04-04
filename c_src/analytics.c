#include "analytics.h"

#include <math.h>
#include <stdatomic.h>
#include <stdint.h>

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#define PM_ANALYTICS_HAS_AVX512_DISPATCH 1
#define PM_ANALYTICS_TARGET_AVX512 __attribute__((target("avx512f")))
#include <immintrin.h>
#else
#define PM_ANALYTICS_HAS_AVX512_DISPATCH 0
#define PM_ANALYTICS_TARGET_AVX512
#endif

#if defined(__GNUC__) || defined(__clang__)
#define ANALYTICS_UNROLL_4 _Pragma("GCC unroll 4")
#else
#define ANALYTICS_UNROLL_4
#endif

static const double ANALYTICS_EPS = 1e-12;
static const double ANALYTICS_ONE_MINUS_EPS = 1.0 - 1e-12;
static const double ANALYTICS_NR_EPS = 1e-9;
static const int ANALYTICS_NR_ITERS = 4;

static _Atomic int analytics_has_avx512f_cache = -1;

static inline double analytics_clamp(double v, double lo, double hi) {
    return fmin(hi, fmax(lo, v));
}

static inline int analytics_runtime_has_avx512f(void) {
    int cached = atomic_load_explicit(&analytics_has_avx512f_cache, memory_order_relaxed);
    if (cached >= 0) {
        return cached;
    }

    int has_avx512f = 0;

#if PM_ANALYTICS_HAS_AVX512_DISPATCH
    __builtin_cpu_init();
    has_avx512f = __builtin_cpu_supports("avx512f") ? 1 : 0;
#endif

    atomic_store_explicit(&analytics_has_avx512f_cache, has_avx512f, memory_order_relaxed);
    return has_avx512f;
}

#if PM_ANALYTICS_HAS_AVX512_DISPATCH
PM_ANALYTICS_TARGET_AVX512
static inline double analytics_reduce_add_pd(__m512d x) {
    double tmp[8];
    _mm512_storeu_pd(tmp, x);
    return tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
}
#endif

void pm_internal_implied_belief_volatility_batch_portable(
    const double *bid_p,
    const double *ask_p,
    const double *q_t,
    const double *gamma,
    const double *tau,
    const double *k,
    double *out_sigma_b,
    size_t n
) {
    if (bid_p == NULL || ask_p == NULL || gamma == NULL || tau == NULL || k == NULL || out_sigma_b == NULL || n == 0u) {
        return;
    }

    (void)q_t;

    ANALYTICS_UNROLL_4
    for (size_t i = 0u; i < n; ++i) {
        const double bid = analytics_clamp(bid_p[i], ANALYTICS_EPS, ANALYTICS_ONE_MINUS_EPS);
        const double ask = analytics_clamp(ask_p[i], ANALYTICS_EPS, ANALYTICS_ONE_MINUS_EPS);
        const double gamma_v = fmax(0.0, gamma[i]);
        const double tau_v = fmax(0.0, tau[i]);
        const double k_v = fmax(ANALYTICS_EPS, k[i]);

        if (ask <= bid || gamma_v <= ANALYTICS_EPS || tau_v <= ANALYTICS_EPS) {
            out_sigma_b[i] = 0.0;
            continue;
        }

        const double target_spread = kernel_logit(ask) - kernel_logit(bid);
        const double two_non_linear = 2.0 * (log1p(gamma_v / k_v) / k_v);
        const double gamma_tau = fmax(ANALYTICS_NR_EPS, gamma_v * tau_v);

        double sigma = sqrt(fmax(0.0, (target_spread - two_non_linear) / gamma_tau));

        for (int iter = 0; iter < ANALYTICS_NR_ITERS; ++iter) {
            const double f = (gamma_tau * sigma * sigma) + two_non_linear - target_spread;
            const double fp = fmax(ANALYTICS_NR_EPS, 2.0 * gamma_tau * sigma);
            sigma = fmax(0.0, sigma - (f / fp));
        }

        out_sigma_b[i] = sigma;
    }
}

#if PM_ANALYTICS_HAS_AVX512_DISPATCH
PM_ANALYTICS_TARGET_AVX512
void pm_internal_implied_belief_volatility_batch_avx512(
    const double *bid_p,
    const double *ask_p,
    const double *q_t,
    const double *gamma,
    const double *tau,
    const double *k,
    double *out_sigma_b,
    size_t n
) {
    if (bid_p == NULL || ask_p == NULL || gamma == NULL || tau == NULL || k == NULL || out_sigma_b == NULL || n == 0u) {
        return;
    }

    (void)q_t;

    const __m512d v_zero = _mm512_set1_pd(0.0);
    const __m512d v_eps = _mm512_set1_pd(ANALYTICS_NR_EPS);
    const __m512d v_two = _mm512_set1_pd(2.0);

    size_t i = 0u;
    for (; (i + 7u) < n; i += 8u) {
        double target_spread_buf[8];
        double two_non_linear_buf[8];
        double gamma_tau_buf[8];
        double sigma_init_buf[8];
        double valid_buf[8];

        for (size_t lane = 0u; lane < 8u; ++lane) {
            const size_t idx = i + lane;
            const double bid = analytics_clamp(bid_p[idx], ANALYTICS_EPS, ANALYTICS_ONE_MINUS_EPS);
            const double ask = analytics_clamp(ask_p[idx], ANALYTICS_EPS, ANALYTICS_ONE_MINUS_EPS);
            const double gamma_v = fmax(0.0, gamma[idx]);
            const double tau_v = fmax(0.0, tau[idx]);
            const double k_v = fmax(ANALYTICS_EPS, k[idx]);
            const double target_spread = kernel_logit(ask) - kernel_logit(bid);
            const double two_non_linear = 2.0 * (log1p(gamma_v / k_v) / k_v);
            const double gamma_tau = fmax(ANALYTICS_NR_EPS, gamma_v * tau_v);

            const double numer = fmax(0.0, target_spread - two_non_linear);
            sigma_init_buf[lane] = sqrt(numer / gamma_tau);
            target_spread_buf[lane] = target_spread;
            two_non_linear_buf[lane] = two_non_linear;
            gamma_tau_buf[lane] = gamma_tau;
            valid_buf[lane] = (ask > bid && gamma_v > ANALYTICS_EPS && tau_v > ANALYTICS_EPS) ? 1.0 : 0.0;
        }

        __m512d sigma = _mm512_loadu_pd(sigma_init_buf);
        const __m512d target_spread = _mm512_loadu_pd(target_spread_buf);
        const __m512d two_non_linear = _mm512_loadu_pd(two_non_linear_buf);
        const __m512d gamma_tau = _mm512_loadu_pd(gamma_tau_buf);

        for (int iter = 0; iter < ANALYTICS_NR_ITERS; ++iter) {
            const __m512d sigma2 = _mm512_mul_pd(sigma, sigma);
            const __m512d f = _mm512_sub_pd(_mm512_add_pd(_mm512_mul_pd(gamma_tau, sigma2), two_non_linear), target_spread);
            const __m512d fp = _mm512_max_pd(v_eps, _mm512_mul_pd(v_two, _mm512_mul_pd(gamma_tau, sigma)));
            const __m512d step = _mm512_div_pd(f, fp);
            sigma = _mm512_max_pd(v_zero, _mm512_sub_pd(sigma, step));
        }

        _mm512_storeu_pd(out_sigma_b + i, sigma);

        for (size_t lane = 0u; lane < 8u; ++lane) {
            if (valid_buf[lane] < 0.5) {
                out_sigma_b[i + lane] = 0.0;
            }
        }
    }

    pm_internal_implied_belief_volatility_batch_portable(
        bid_p + i,
        ask_p + i,
        q_t != NULL ? q_t + i : NULL,
        gamma + i,
        tau + i,
        k + i,
        out_sigma_b + i,
        n - i
    );
}
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
) {
    if (analytics_runtime_has_avx512f()) {
#if PM_ANALYTICS_HAS_AVX512_DISPATCH
        pm_internal_implied_belief_volatility_batch_avx512(
            bid_p,
            ask_p,
            q_t,
            gamma,
            tau,
            k,
            out_sigma_b,
            n
        );
        return;
#endif
    }

    pm_internal_implied_belief_volatility_batch_portable(
        bid_p,
        ask_p,
        q_t,
        gamma,
        tau,
        k,
        out_sigma_b,
        n
    );
}

void pm_internal_simulate_shock_logit_batch_portable(
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
) {
    if (x_t == NULL || q_t == NULL || sigma_b == NULL || gamma == NULL || tau == NULL || k == NULL || shock_p == NULL ||
        out_r_x == NULL || out_bid_p == NULL || out_ask_p == NULL || out_greeks == NULL || out_pnl_shift == NULL || n == 0u) {
        return;
    }

    ANALYTICS_UNROLL_4
    for (size_t i = 0u; i < n; ++i) {
        const double base_p = kernel_sigmoid(x_t[i]);
        const double shocked_p = analytics_clamp(base_p + shock_p[i], ANALYTICS_EPS, ANALYTICS_ONE_MINUS_EPS);
        const double x_shocked = kernel_logit(shocked_p);

        const double gamma_v = fmax(0.0, gamma[i]);
        const double tau_v = fmax(0.0, tau[i]);
        const double k_v = fmax(ANALYTICS_EPS, k[i]);

        const double sigma2 = sigma_b[i] * sigma_b[i];
        const double risk_term = gamma_v * sigma2 * tau_v;
        const double r_x = x_shocked - (q_t[i] * risk_term);
        const double half_spread = (0.5 * risk_term) + (log1p(gamma_v / k_v) / k_v);

        out_r_x[i] = r_x;
        out_bid_p[i] = kernel_sigmoid(r_x - half_spread);
        out_ask_p[i] = kernel_sigmoid(r_x + half_spread);

        kernel_greeks_from_logit(x_shocked, &out_greeks[i].delta_x, &out_greeks[i].gamma_x);
        out_pnl_shift[i] = q_t[i] * (shocked_p - base_p);
    }
}

#if PM_ANALYTICS_HAS_AVX512_DISPATCH
PM_ANALYTICS_TARGET_AVX512
void pm_internal_simulate_shock_logit_batch_avx512(
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
) {
    if (x_t == NULL || q_t == NULL || sigma_b == NULL || gamma == NULL || tau == NULL || k == NULL || shock_p == NULL ||
        out_r_x == NULL || out_bid_p == NULL || out_ask_p == NULL || out_greeks == NULL || out_pnl_shift == NULL || n == 0u) {
        return;
    }

    const __m512d v_zero = _mm512_set1_pd(0.0);
    const __m512d v_half = _mm512_set1_pd(0.5);

    size_t i = 0u;
    for (; (i + 7u) < n; i += 8u) {
        double x_shocked_buf[8];
        double pnl_shift_buf[8];
        double non_linear_buf[8];
        double bid_x_buf[8];
        double ask_x_buf[8];
        double bid_p_buf[8];
        double ask_p_buf[8];

        for (size_t lane = 0u; lane < 8u; ++lane) {
            const size_t idx = i + lane;
            const double base_p = kernel_sigmoid(x_t[idx]);
            const double shocked_p = analytics_clamp(base_p + shock_p[idx], ANALYTICS_EPS, ANALYTICS_ONE_MINUS_EPS);

            x_shocked_buf[lane] = kernel_logit(shocked_p);
            pnl_shift_buf[lane] = q_t[idx] * (shocked_p - base_p);

            const double gamma_v = fmax(0.0, gamma[idx]);
            const double k_v = fmax(ANALYTICS_EPS, k[idx]);
            non_linear_buf[lane] = log1p(gamma_v / k_v) / k_v;
        }

        const __m512d x_shocked = _mm512_loadu_pd(x_shocked_buf);
        const __m512d q = _mm512_loadu_pd(q_t + i);
        const __m512d sigma = _mm512_loadu_pd(sigma_b + i);
        const __m512d gamma_raw = _mm512_loadu_pd(gamma + i);
        const __m512d tau_raw = _mm512_loadu_pd(tau + i);
        const __m512d non_linear = _mm512_loadu_pd(non_linear_buf);

        const __m512d gamma_v = _mm512_max_pd(v_zero, gamma_raw);
        const __m512d tau_v = _mm512_max_pd(v_zero, tau_raw);
        const __m512d sigma2 = _mm512_mul_pd(sigma, sigma);
        const __m512d risk_term = _mm512_mul_pd(_mm512_mul_pd(gamma_v, sigma2), tau_v);

        const __m512d r_x = _mm512_sub_pd(x_shocked, _mm512_mul_pd(q, risk_term));
        const __m512d half_spread = _mm512_add_pd(_mm512_mul_pd(v_half, risk_term), non_linear);

        const __m512d bid_x = _mm512_sub_pd(r_x, half_spread);
        const __m512d ask_x = _mm512_add_pd(r_x, half_spread);

        _mm512_storeu_pd(out_r_x + i, r_x);
        _mm512_storeu_pd(bid_x_buf, bid_x);
        _mm512_storeu_pd(ask_x_buf, ask_x);

        kernel_sigmoid_batch(bid_x_buf, bid_p_buf, 8u);
        kernel_sigmoid_batch(ask_x_buf, ask_p_buf, 8u);
        kernel_greeks_batch(x_shocked_buf, out_greeks + i, 8u);

        for (size_t lane = 0u; lane < 8u; ++lane) {
            out_bid_p[i + lane] = bid_p_buf[lane];
            out_ask_p[i + lane] = ask_p_buf[lane];
            out_pnl_shift[i + lane] = pnl_shift_buf[lane];
        }
    }

    pm_internal_simulate_shock_logit_batch_portable(
        x_t + i,
        q_t + i,
        sigma_b + i,
        gamma + i,
        tau + i,
        k + i,
        shock_p + i,
        out_r_x + i,
        out_bid_p + i,
        out_ask_p + i,
        out_greeks + i,
        out_pnl_shift + i,
        n - i
    );
}
#endif

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
) {
    if (analytics_runtime_has_avx512f()) {
#if PM_ANALYTICS_HAS_AVX512_DISPATCH
        pm_internal_simulate_shock_logit_batch_avx512(
            x_t,
            q_t,
            sigma_b,
            gamma,
            tau,
            k,
            shock_p,
            out_r_x,
            out_bid_p,
            out_ask_p,
            out_greeks,
            out_pnl_shift,
            n
        );
        return;
#endif
    }

    pm_internal_simulate_shock_logit_batch_portable(
        x_t,
        q_t,
        sigma_b,
        gamma,
        tau,
        k,
        shock_p,
        out_r_x,
        out_bid_p,
        out_ask_p,
        out_greeks,
        out_pnl_shift,
        n
    );
}

void pm_internal_adaptive_kelly_clip_batch_portable(
    const double *belief_p,
    const double *market_p,
    const double *q_t,
    const double *gamma,
    const double *risk_limit,
    const double *max_clip,
    double *out_maker_clip,
    double *out_taker_clip,
    size_t n
) {
    if (belief_p == NULL || market_p == NULL || q_t == NULL || gamma == NULL || risk_limit == NULL || max_clip == NULL ||
        out_maker_clip == NULL || out_taker_clip == NULL || n == 0u) {
        return;
    }

    ANALYTICS_UNROLL_4
    for (size_t i = 0u; i < n; ++i) {
        const double belief = analytics_clamp(belief_p[i], ANALYTICS_EPS, ANALYTICS_ONE_MINUS_EPS);
        const double market = analytics_clamp(market_p[i], ANALYTICS_EPS, ANALYTICS_ONE_MINUS_EPS);
        const double edge = belief - market;
        const double variance = fmax(ANALYTICS_EPS, market * (1.0 - market));
        const double kelly_frac = edge / variance;

        const double gamma_v = fmax(0.0, gamma[i]);
        const double inventory_scale = 1.0 / (1.0 + (gamma_v * fabs(q_t[i])));

        const double risk = fmax(ANALYTICS_EPS, risk_limit[i]);
        const double clip_cap = fmax(ANALYTICS_EPS, max_clip[i]);

        double taker = kelly_frac * risk * inventory_scale;
        taker = analytics_clamp(taker, -clip_cap, clip_cap);

        const double long_limit = risk - q_t[i];
        const double short_limit = -risk - q_t[i];

        taker = analytics_clamp(taker, short_limit, long_limit);
        const double maker = analytics_clamp(0.5 * taker, short_limit, long_limit);

        out_maker_clip[i] = maker;
        out_taker_clip[i] = taker;
    }
}

#if PM_ANALYTICS_HAS_AVX512_DISPATCH
PM_ANALYTICS_TARGET_AVX512
void pm_internal_adaptive_kelly_clip_batch_avx512(
    const double *belief_p,
    const double *market_p,
    const double *q_t,
    const double *gamma,
    const double *risk_limit,
    const double *max_clip,
    double *out_maker_clip,
    double *out_taker_clip,
    size_t n
) {
    if (belief_p == NULL || market_p == NULL || q_t == NULL || gamma == NULL || risk_limit == NULL || max_clip == NULL ||
        out_maker_clip == NULL || out_taker_clip == NULL || n == 0u) {
        return;
    }

    const __m512d v_zero = _mm512_set1_pd(0.0);
    const __m512d v_one = _mm512_set1_pd(1.0);
    const __m512d v_half = _mm512_set1_pd(0.5);
    const __m512d v_eps = _mm512_set1_pd(ANALYTICS_EPS);
    const __m512d v_one_minus_eps = _mm512_set1_pd(ANALYTICS_ONE_MINUS_EPS);
    const __m512d v_abs_mask = _mm512_castsi512_pd(_mm512_set1_epi64(0x7fffffffffffffffULL));

    size_t i = 0u;
    for (; (i + 7u) < n; i += 8u) {
        const __m512d belief_raw = _mm512_loadu_pd(belief_p + i);
        const __m512d market_raw = _mm512_loadu_pd(market_p + i);
        const __m512d q = _mm512_loadu_pd(q_t + i);
        const __m512d gamma_raw = _mm512_loadu_pd(gamma + i);
        const __m512d risk_raw = _mm512_loadu_pd(risk_limit + i);
        const __m512d clip_raw = _mm512_loadu_pd(max_clip + i);

        const __m512d belief = _mm512_max_pd(v_eps, _mm512_min_pd(v_one_minus_eps, belief_raw));
        const __m512d market = _mm512_max_pd(v_eps, _mm512_min_pd(v_one_minus_eps, market_raw));

        const __m512d edge = _mm512_sub_pd(belief, market);
        const __m512d variance = _mm512_max_pd(v_eps, _mm512_mul_pd(market, _mm512_sub_pd(v_one, market)));
        const __m512d kelly_frac = _mm512_div_pd(edge, variance);

        const __m512d gamma_v = _mm512_max_pd(v_zero, gamma_raw);
        const __m512d abs_q = _mm512_and_pd(q, v_abs_mask);
        const __m512d inventory_scale = _mm512_div_pd(v_one, _mm512_add_pd(v_one, _mm512_mul_pd(gamma_v, abs_q)));

        const __m512d risk = _mm512_max_pd(v_eps, risk_raw);
        const __m512d clip_cap = _mm512_max_pd(v_eps, clip_raw);
        const __m512d neg_clip_cap = _mm512_sub_pd(v_zero, clip_cap);

        __m512d taker = _mm512_mul_pd(kelly_frac, _mm512_mul_pd(risk, inventory_scale));
        taker = _mm512_max_pd(neg_clip_cap, _mm512_min_pd(clip_cap, taker));

        const __m512d long_limit = _mm512_sub_pd(risk, q);
        const __m512d short_limit = _mm512_sub_pd(_mm512_sub_pd(v_zero, risk), q);
        taker = _mm512_max_pd(short_limit, _mm512_min_pd(long_limit, taker));

        __m512d maker = _mm512_mul_pd(v_half, taker);
        maker = _mm512_max_pd(short_limit, _mm512_min_pd(long_limit, maker));

        _mm512_storeu_pd(out_maker_clip + i, maker);
        _mm512_storeu_pd(out_taker_clip + i, taker);
    }

    pm_internal_adaptive_kelly_clip_batch_portable(
        belief_p + i,
        market_p + i,
        q_t + i,
        gamma + i,
        risk_limit + i,
        max_clip + i,
        out_maker_clip + i,
        out_taker_clip + i,
        n - i
    );
}
#endif

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
) {
    if (analytics_runtime_has_avx512f()) {
#if PM_ANALYTICS_HAS_AVX512_DISPATCH
        pm_internal_adaptive_kelly_clip_batch_avx512(
            belief_p,
            market_p,
            q_t,
            gamma,
            risk_limit,
            max_clip,
            out_maker_clip,
            out_taker_clip,
            n
        );
        return;
#endif
    }

    pm_internal_adaptive_kelly_clip_batch_portable(
        belief_p,
        market_p,
        q_t,
        gamma,
        risk_limit,
        max_clip,
        out_maker_clip,
        out_taker_clip,
        n
    );
}

void pm_internal_order_book_microstructure_batch_portable(
    const double *bid_p,
    const double *ask_p,
    const double *bid_vol,
    const double *ask_vol,
    double *out_obi,
    double *out_vwm_p,
    double *out_vwm_x,
    double *out_pressure,
    size_t n
) {
    if (bid_p == NULL || ask_p == NULL || bid_vol == NULL || ask_vol == NULL || out_obi == NULL || out_vwm_p == NULL ||
        out_vwm_x == NULL || out_pressure == NULL || n == 0u) {
        return;
    }

    ANALYTICS_UNROLL_4
    for (size_t i = 0u; i < n; ++i) {
        const double bid = analytics_clamp(bid_p[i], ANALYTICS_EPS, ANALYTICS_ONE_MINUS_EPS);
        const double ask = analytics_clamp(ask_p[i], ANALYTICS_EPS, ANALYTICS_ONE_MINUS_EPS);
        const double bv = fmax(0.0, bid_vol[i]);
        const double av = fmax(0.0, ask_vol[i]);

        const double vol_sum = fmax(ANALYTICS_EPS, bv + av);
        const double obi = (bv - av) / vol_sum;

        const double vwm = analytics_clamp(((ask * bv) + (bid * av)) / vol_sum, ANALYTICS_EPS, ANALYTICS_ONE_MINUS_EPS);
        const double mid = 0.5 * (bid + ask);
        const double spread = fmax(ANALYTICS_EPS, ask - bid);
        const double pressure = obi + ((vwm - mid) / spread);

        out_obi[i] = obi;
        out_vwm_p[i] = vwm;
        out_vwm_x[i] = kernel_logit(vwm);
        out_pressure[i] = pressure;
    }
}

#if PM_ANALYTICS_HAS_AVX512_DISPATCH
PM_ANALYTICS_TARGET_AVX512
void pm_internal_order_book_microstructure_batch_avx512(
    const double *bid_p,
    const double *ask_p,
    const double *bid_vol,
    const double *ask_vol,
    double *out_obi,
    double *out_vwm_p,
    double *out_vwm_x,
    double *out_pressure,
    size_t n
) {
    if (bid_p == NULL || ask_p == NULL || bid_vol == NULL || ask_vol == NULL || out_obi == NULL || out_vwm_p == NULL ||
        out_vwm_x == NULL || out_pressure == NULL || n == 0u) {
        return;
    }

    const __m512d v_eps = _mm512_set1_pd(ANALYTICS_EPS);
    const __m512d v_half = _mm512_set1_pd(0.5);
    const __m512d v_one_minus_eps = _mm512_set1_pd(ANALYTICS_ONE_MINUS_EPS);
    const __m512d v_zero = _mm512_set1_pd(0.0);

    size_t i = 0u;
    for (; (i + 7u) < n; i += 8u) {
        double bid_buf[8];
        double ask_buf[8];
        double bv_buf[8];
        double av_buf[8];

        for (size_t lane = 0u; lane < 8u; ++lane) {
            const size_t idx = i + lane;
            bid_buf[lane] = analytics_clamp(bid_p[idx], ANALYTICS_EPS, ANALYTICS_ONE_MINUS_EPS);
            ask_buf[lane] = analytics_clamp(ask_p[idx], ANALYTICS_EPS, ANALYTICS_ONE_MINUS_EPS);
            bv_buf[lane] = fmax(0.0, bid_vol[idx]);
            av_buf[lane] = fmax(0.0, ask_vol[idx]);
        }

        const __m512d bid = _mm512_loadu_pd(bid_buf);
        const __m512d ask = _mm512_loadu_pd(ask_buf);
        const __m512d bv = _mm512_loadu_pd(bv_buf);
        const __m512d av = _mm512_loadu_pd(av_buf);

        const __m512d vol_sum = _mm512_max_pd(v_eps, _mm512_add_pd(bv, av));
        const __m512d obi = _mm512_div_pd(_mm512_sub_pd(bv, av), vol_sum);

        const __m512d vwm_num = _mm512_add_pd(_mm512_mul_pd(ask, bv), _mm512_mul_pd(bid, av));
        const __m512d vwm = _mm512_max_pd(v_eps, _mm512_min_pd(v_one_minus_eps, _mm512_div_pd(vwm_num, vol_sum)));

        const __m512d mid = _mm512_mul_pd(v_half, _mm512_add_pd(bid, ask));
        const __m512d raw_spread = _mm512_sub_pd(ask, bid);
        const __m512d spread = _mm512_max_pd(v_eps, _mm512_max_pd(v_zero, raw_spread));
        const __m512d skew = _mm512_div_pd(_mm512_sub_pd(vwm, mid), spread);
        const __m512d pressure = _mm512_add_pd(obi, skew);

        double vwm_buf[8];
        double vwm_x_buf[8];

        _mm512_storeu_pd(out_obi + i, obi);
        _mm512_storeu_pd(out_vwm_p + i, vwm);
        _mm512_storeu_pd(out_pressure + i, pressure);
        _mm512_storeu_pd(vwm_buf, vwm);

        kernel_logit_batch(vwm_buf, vwm_x_buf, 8u);
        for (size_t lane = 0u; lane < 8u; ++lane) {
            out_vwm_x[i + lane] = vwm_x_buf[lane];
        }
    }

    pm_internal_order_book_microstructure_batch_portable(
        bid_p + i,
        ask_p + i,
        bid_vol + i,
        ask_vol + i,
        out_obi + i,
        out_vwm_p + i,
        out_vwm_x + i,
        out_pressure + i,
        n - i
    );
}
#endif

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
) {
    if (analytics_runtime_has_avx512f()) {
#if PM_ANALYTICS_HAS_AVX512_DISPATCH
        pm_internal_order_book_microstructure_batch_avx512(
            bid_p,
            ask_p,
            bid_vol,
            ask_vol,
            out_obi,
            out_vwm_p,
            out_vwm_x,
            out_pressure,
            n
        );
        return;
#endif
    }

    pm_internal_order_book_microstructure_batch_portable(
        bid_p,
        ask_p,
        bid_vol,
        ask_vol,
        out_obi,
        out_vwm_p,
        out_vwm_x,
        out_pressure,
        n
    );
}

void pm_internal_aggregate_portfolio_greeks_portable(
    const double *positions,
    const double *delta_x,
    const double *gamma_x,
    const double *weights,
    const double *corr_matrix,
    size_t n,
    double *out_net_delta,
    double *out_net_gamma
) {
    if (positions == NULL || delta_x == NULL || gamma_x == NULL || out_net_delta == NULL || out_net_gamma == NULL) {
        return;
    }

    if (n == 0u) {
        *out_net_delta = 0.0;
        *out_net_gamma = 0.0;
        return;
    }

    double net_delta = 0.0;
    double net_gamma = 0.0;

    if (corr_matrix == NULL) {
        ANALYTICS_UNROLL_4
        for (size_t i = 0u; i < n; ++i) {
            const double w = (weights != NULL) ? weights[i] : 1.0;
            net_delta += positions[i] * delta_x[i] * w;
            net_gamma += positions[i] * gamma_x[i] * w;
        }

        *out_net_delta = net_delta;
        *out_net_gamma = net_gamma;
        return;
    }

    for (size_t i = 0u; i < n; ++i) {
        const double wi = (weights != NULL) ? weights[i] : 1.0;
        const double exp_delta_i = positions[i] * delta_x[i] * wi;
        const double exp_gamma_i = positions[i] * gamma_x[i] * wi;

        const double *corr_row = corr_matrix + (i * n);
        double row_delta = 0.0;
        double row_gamma = 0.0;

        ANALYTICS_UNROLL_4
        for (size_t j = 0u; j < n; ++j) {
            const double wj = (weights != NULL) ? weights[j] : 1.0;
            const double exp_delta_j = positions[j] * delta_x[j] * wj;
            const double exp_gamma_j = positions[j] * gamma_x[j] * wj;
            const double corr = corr_row[j];

            row_delta += corr * exp_delta_j;
            row_gamma += corr * exp_gamma_j;
        }

        net_delta += exp_delta_i * row_delta;
        net_gamma += exp_gamma_i * row_gamma;
    }

    *out_net_delta = net_delta;
    *out_net_gamma = net_gamma;
}

#if PM_ANALYTICS_HAS_AVX512_DISPATCH
PM_ANALYTICS_TARGET_AVX512
void pm_internal_aggregate_portfolio_greeks_avx512(
    const double *positions,
    const double *delta_x,
    const double *gamma_x,
    const double *weights,
    const double *corr_matrix,
    size_t n,
    double *out_net_delta,
    double *out_net_gamma
) {
    if (positions == NULL || delta_x == NULL || gamma_x == NULL || out_net_delta == NULL || out_net_gamma == NULL) {
        return;
    }

    if (n == 0u) {
        *out_net_delta = 0.0;
        *out_net_gamma = 0.0;
        return;
    }

    double net_delta = 0.0;
    double net_gamma = 0.0;

    if (corr_matrix == NULL) {
        size_t i = 0u;
        __m512d acc_delta = _mm512_setzero_pd();
        __m512d acc_gamma = _mm512_setzero_pd();

        for (; (i + 7u) < n; i += 8u) {
            const __m512d pos = _mm512_loadu_pd(positions + i);
            const __m512d del = _mm512_loadu_pd(delta_x + i);
            const __m512d gam = _mm512_loadu_pd(gamma_x + i);
            const __m512d w = (weights != NULL) ? _mm512_loadu_pd(weights + i) : _mm512_set1_pd(1.0);

            acc_delta = _mm512_add_pd(acc_delta, _mm512_mul_pd(_mm512_mul_pd(pos, del), w));
            acc_gamma = _mm512_add_pd(acc_gamma, _mm512_mul_pd(_mm512_mul_pd(pos, gam), w));
        }

        net_delta += analytics_reduce_add_pd(acc_delta);
        net_gamma += analytics_reduce_add_pd(acc_gamma);

        pm_internal_aggregate_portfolio_greeks_portable(
            positions + i,
            delta_x + i,
            gamma_x + i,
            weights != NULL ? weights + i : NULL,
            NULL,
            n - i,
            out_net_delta,
            out_net_gamma
        );

        net_delta += *out_net_delta;
        net_gamma += *out_net_gamma;
        *out_net_delta = net_delta;
        *out_net_gamma = net_gamma;
        return;
    }

    for (size_t i = 0u; i < n; ++i) {
        const double wi = (weights != NULL) ? weights[i] : 1.0;
        const double exp_delta_i = positions[i] * delta_x[i] * wi;
        const double exp_gamma_i = positions[i] * gamma_x[i] * wi;

        const double *corr_row = corr_matrix + (i * n);
        double row_delta = 0.0;
        double row_gamma = 0.0;

        size_t j = 0u;
        __m512d acc_delta = _mm512_setzero_pd();
        __m512d acc_gamma = _mm512_setzero_pd();

        for (; (j + 7u) < n; j += 8u) {
            const __m512d corr = _mm512_loadu_pd(corr_row + j);
            const __m512d pos = _mm512_loadu_pd(positions + j);
            const __m512d del = _mm512_loadu_pd(delta_x + j);
            const __m512d gam = _mm512_loadu_pd(gamma_x + j);
            const __m512d wj = (weights != NULL) ? _mm512_loadu_pd(weights + j) : _mm512_set1_pd(1.0);

            const __m512d exp_delta_j = _mm512_mul_pd(_mm512_mul_pd(pos, del), wj);
            const __m512d exp_gamma_j = _mm512_mul_pd(_mm512_mul_pd(pos, gam), wj);

            acc_delta = _mm512_add_pd(acc_delta, _mm512_mul_pd(corr, exp_delta_j));
            acc_gamma = _mm512_add_pd(acc_gamma, _mm512_mul_pd(corr, exp_gamma_j));
        }

        row_delta += analytics_reduce_add_pd(acc_delta);
        row_gamma += analytics_reduce_add_pd(acc_gamma);

        for (; j < n; ++j) {
            const double wj = (weights != NULL) ? weights[j] : 1.0;
            const double exp_delta_j = positions[j] * delta_x[j] * wj;
            const double exp_gamma_j = positions[j] * gamma_x[j] * wj;
            const double corr = corr_row[j];

            row_delta += corr * exp_delta_j;
            row_gamma += corr * exp_gamma_j;
        }

        net_delta += exp_delta_i * row_delta;
        net_gamma += exp_gamma_i * row_gamma;
    }

    *out_net_delta = net_delta;
    *out_net_gamma = net_gamma;
}
#endif

void aggregate_portfolio_greeks(
    const double *positions,
    const double *delta_x,
    const double *gamma_x,
    const double *weights,
    const double *corr_matrix,
    size_t n,
    double *out_net_delta,
    double *out_net_gamma
) {
    if (analytics_runtime_has_avx512f()) {
#if PM_ANALYTICS_HAS_AVX512_DISPATCH
        pm_internal_aggregate_portfolio_greeks_avx512(
            positions,
            delta_x,
            gamma_x,
            weights,
            corr_matrix,
            n,
            out_net_delta,
            out_net_gamma
        );
        return;
#endif
    }

    pm_internal_aggregate_portfolio_greeks_portable(
        positions,
        delta_x,
        gamma_x,
        weights,
        corr_matrix,
        n,
        out_net_delta,
        out_net_gamma
    );
}
