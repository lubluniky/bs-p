#include "kernel.h"

#include <math.h>
#include <stdint.h>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

#if defined(__GNUC__) || defined(__clang__)
#define KERNEL_UNROLL_4 _Pragma("GCC unroll 4")
#else
#define KERNEL_UNROLL_4
#endif

#ifndef KERNEL_USE_FAST_SIGMOID
#define KERNEL_USE_FAST_SIGMOID 1
#endif

static const double KERNEL_EPS = 1e-12;
static const double KERNEL_ONE_MINUS_EPS = 1.0 - 1e-12;
static const double KERNEL_SIGMOID_CLIP = 10.0;

_Static_assert(sizeof(market_state_t) == 64u, "market_state_t must be exactly 64 bytes");

static inline double kernel_clamp(double v, double lo, double hi) {
    return fmin(hi, fmax(lo, v));
}

/*
 * 7/7 Pade-like tanh approximation mapped through sigmoid:
 * sigmoid(x) = 0.5 * (1 + tanh(x / 2))
 * We clip x to [-10, 10] to keep approximation error bounded and stable.
 */
static inline double kernel_sigmoid_fast(double x) {
    const double xc = kernel_clamp(x, -KERNEL_SIGMOID_CLIP, KERNEL_SIGMOID_CLIP);
    const double z = 0.5 * xc;
    const double z2 = z * z;

    const double num = z * (135135.0 + z2 * (17325.0 + z2 * (378.0 + z2)));
    const double den = 135135.0 + z2 * (62370.0 + z2 * (3150.0 + z2 * 28.0));

    return kernel_clamp(0.5 * (1.0 + (num / den)), KERNEL_EPS, KERNEL_ONE_MINUS_EPS);
}

#if !KERNEL_USE_FAST_SIGMOID
static inline double kernel_sigmoid_exact(double x) {
    const double e = exp(-fabs(x));
    const double inv = 1.0 / (1.0 + e);
    const double p = (x >= 0.0) ? inv : (e * inv);
    return kernel_clamp(p, KERNEL_EPS, KERNEL_ONE_MINUS_EPS);
}
#endif

static inline double kernel_sigmoid_internal(double x) {
#if KERNEL_USE_FAST_SIGMOID
    return kernel_sigmoid_fast(x);
#else
    return kernel_sigmoid_exact(x);
#endif
}

#if defined(__AVX512F__)
static inline __m512d kernel_sigmoid_fast_avx512(__m512d x) {
    const __m512d v_clip_hi = _mm512_set1_pd(KERNEL_SIGMOID_CLIP);
    const __m512d v_clip_lo = _mm512_set1_pd(-KERNEL_SIGMOID_CLIP);
    const __m512d v_half = _mm512_set1_pd(0.5);
    const __m512d v_one = _mm512_set1_pd(1.0);
    const __m512d v_eps = _mm512_set1_pd(KERNEL_EPS);
    const __m512d v_one_minus_eps = _mm512_set1_pd(KERNEL_ONE_MINUS_EPS);

    const __m512d a0 = _mm512_set1_pd(135135.0);
    const __m512d a1 = _mm512_set1_pd(17325.0);
    const __m512d a2 = _mm512_set1_pd(378.0);
    const __m512d b1 = _mm512_set1_pd(62370.0);
    const __m512d b2 = _mm512_set1_pd(3150.0);
    const __m512d b3 = _mm512_set1_pd(28.0);

    const __m512d xc = _mm512_max_pd(v_clip_lo, _mm512_min_pd(v_clip_hi, x));
    const __m512d z = _mm512_mul_pd(v_half, xc);
    const __m512d z2 = _mm512_mul_pd(z, z);

    const __m512d t_num_0 = _mm512_add_pd(a2, z2);
    const __m512d t_num_1 = _mm512_add_pd(a1, _mm512_mul_pd(z2, t_num_0));
    const __m512d t_num_2 = _mm512_add_pd(a0, _mm512_mul_pd(z2, t_num_1));
    const __m512d num = _mm512_mul_pd(z, t_num_2);

    const __m512d t_den_0 = _mm512_add_pd(b2, _mm512_mul_pd(b3, z2));
    const __m512d t_den_1 = _mm512_add_pd(b1, _mm512_mul_pd(z2, t_den_0));
    const __m512d den = _mm512_add_pd(a0, _mm512_mul_pd(z2, t_den_1));

    const __m512d tanh_approx = _mm512_div_pd(num, den);
    const __m512d p = _mm512_mul_pd(v_half, _mm512_add_pd(v_one, tanh_approx));

    return _mm512_max_pd(v_eps, _mm512_min_pd(v_one_minus_eps, p));
}
#endif

double kernel_sigmoid(double x) {
    return kernel_sigmoid_internal(x);
}

double kernel_logit(double p) {
    const double pc = kernel_clamp(p, KERNEL_EPS, KERNEL_ONE_MINUS_EPS);
    return log(pc / (1.0 - pc));
}

void kernel_sigmoid_batch(const double *x, double *out_p, size_t n) {
    if (x == NULL || out_p == NULL || n == 0u) {
        return;
    }

    size_t i = 0u;

#if defined(__AVX512F__) && KERNEL_USE_FAST_SIGMOID
    for (; (i + 7u) < n; i += 8u) {
        const __m512d vx = _mm512_loadu_pd(x + i);
        const __m512d vp = kernel_sigmoid_fast_avx512(vx);
        _mm512_storeu_pd(out_p + i, vp);
    }
#endif

    KERNEL_UNROLL_4
    for (; i < n; ++i) {
        out_p[i] = kernel_sigmoid_internal(x[i]);
    }
}

void kernel_logit_batch(const double *p, double *out_x, size_t n) {
    if (p == NULL || out_x == NULL || n == 0u) {
        return;
    }

    KERNEL_UNROLL_4
    for (size_t i = 0u; i < n; ++i) {
        out_x[i] = kernel_logit(p[i]);
    }
}

void kernel_greeks_from_logit(double x, double *delta_x, double *gamma_x) {
    if (delta_x == NULL || gamma_x == NULL) {
        return;
    }

    const double p = kernel_sigmoid_internal(x);
    const double delta = p * (1.0 - p);
    const double gamma = delta * (1.0 - (2.0 * p));

    *delta_x = delta;
    *gamma_x = gamma;
}

void kernel_greeks_batch(const double *x, greek_out_t *out, size_t n) {
    if (x == NULL || out == NULL || n == 0u) {
        return;
    }

    size_t i = 0u;

#if defined(__AVX512F__) && KERNEL_USE_FAST_SIGMOID
    for (; (i + 7u) < n; i += 8u) {
        const __m512d vx = _mm512_loadu_pd(x + i);
        const __m512d p = kernel_sigmoid_fast_avx512(vx);
        const __m512d one = _mm512_set1_pd(1.0);
        const __m512d two = _mm512_set1_pd(2.0);

        const __m512d delta = _mm512_mul_pd(p, _mm512_sub_pd(one, p));
        const __m512d gamma = _mm512_mul_pd(delta, _mm512_sub_pd(one, _mm512_mul_pd(two, p)));

        double delta_buf[8];
        double gamma_buf[8];
        _mm512_storeu_pd(delta_buf, delta);
        _mm512_storeu_pd(gamma_buf, gamma);

        for (size_t lane = 0u; lane < 8u; ++lane) {
            out[i + lane].delta_x = delta_buf[lane];
            out[i + lane].gamma_x = gamma_buf[lane];
        }
    }
#endif

    KERNEL_UNROLL_4
    for (; i < n; ++i) {
        kernel_greeks_from_logit(x[i], &out[i].delta_x, &out[i].gamma_x);
    }
}

void calculate_quotes_logit(
    const market_state_t *states,
    quote_out_t *quotes,
    size_t n
) {
    if (states == NULL || quotes == NULL || n == 0u) {
        return;
    }

    size_t i = 0u;

#if defined(__AVX512F__)
    {
        const double *state_base = (const double *)states;
        const __m512d v_zero = _mm512_set1_pd(0.0);
        const __m512d v_half = _mm512_set1_pd(0.5);
        const __m512d v_eps = _mm512_set1_pd(KERNEL_EPS);

        const __m512i lane_offsets = _mm512_set_epi64(56, 48, 40, 32, 24, 16, 8, 0);
        const __m512i off_x = _mm512_set1_epi64(0);
        const __m512i off_q = _mm512_set1_epi64(1);
        const __m512i off_sigma = _mm512_set1_epi64(2);
        const __m512i off_gamma = _mm512_set1_epi64(3);
        const __m512i off_tau = _mm512_set1_epi64(4);
        const __m512i off_k = _mm512_set1_epi64(5);

        for (; (i + 7u) < n; i += 8u) {
            const __m512i base_idx = _mm512_set1_epi64((long long)(i * 8u));
            const __m512i idx = _mm512_add_epi64(base_idx, lane_offsets);

            const __m512d x = _mm512_i64gather_pd(_mm512_add_epi64(idx, off_x), state_base, 8);
            const __m512d q = _mm512_i64gather_pd(_mm512_add_epi64(idx, off_q), state_base, 8);
            const __m512d sigma = _mm512_i64gather_pd(_mm512_add_epi64(idx, off_sigma), state_base, 8);
            const __m512d gamma_raw = _mm512_i64gather_pd(_mm512_add_epi64(idx, off_gamma), state_base, 8);
            const __m512d tau_raw = _mm512_i64gather_pd(_mm512_add_epi64(idx, off_tau), state_base, 8);
            const __m512d k_raw = _mm512_i64gather_pd(_mm512_add_epi64(idx, off_k), state_base, 8);

            const __m512d gamma = _mm512_max_pd(v_zero, gamma_raw);
            const __m512d tau = _mm512_max_pd(v_zero, tau_raw);
            const __m512d k = _mm512_max_pd(v_eps, k_raw);

            const __m512d sigma2 = _mm512_mul_pd(sigma, sigma);
            const __m512d variance_horizon = _mm512_mul_pd(sigma2, tau);
            const __m512d risk_term = _mm512_mul_pd(gamma, variance_horizon);

            const __m512d r_x = _mm512_sub_pd(x, _mm512_mul_pd(q, risk_term));

            const __m512d gamma_over_k = _mm512_div_pd(gamma, k);
            const __m512d linear_half_spread = _mm512_mul_pd(v_half, risk_term);

            double gok_buf[8];
            double k_buf[8];
            double non_linear_buf[8];
            _mm512_storeu_pd(gok_buf, gamma_over_k);
            _mm512_storeu_pd(k_buf, k);

            for (size_t lane = 0u; lane < 8u; ++lane) {
                non_linear_buf[lane] = log1p(gok_buf[lane]) / k_buf[lane];
            }

            const __m512d non_linear = _mm512_loadu_pd(non_linear_buf);
            const __m512d delta_x = _mm512_add_pd(linear_half_spread, non_linear);

            const __m512d x_bid = _mm512_sub_pd(r_x, delta_x);
            const __m512d x_ask = _mm512_add_pd(r_x, delta_x);

#if KERNEL_USE_FAST_SIGMOID
            const __m512d bid_p = kernel_sigmoid_fast_avx512(x_bid);
            const __m512d ask_p = kernel_sigmoid_fast_avx512(x_ask);
#else
            double x_bid_buf[8];
            double x_ask_buf[8];
            double bid_buf_tmp[8];
            double ask_buf_tmp[8];
            _mm512_storeu_pd(x_bid_buf, x_bid);
            _mm512_storeu_pd(x_ask_buf, x_ask);
            for (size_t lane = 0u; lane < 8u; ++lane) {
                bid_buf_tmp[lane] = kernel_sigmoid_exact(x_bid_buf[lane]);
                ask_buf_tmp[lane] = kernel_sigmoid_exact(x_ask_buf[lane]);
            }
            const __m512d bid_p = _mm512_loadu_pd(bid_buf_tmp);
            const __m512d ask_p = _mm512_loadu_pd(ask_buf_tmp);
#endif

            double bid_buf[8];
            double ask_buf[8];
            _mm512_storeu_pd(bid_buf, bid_p);
            _mm512_storeu_pd(ask_buf, ask_p);

            for (size_t lane = 0u; lane < 8u; ++lane) {
                quotes[i + lane].bid_p = kernel_clamp(bid_buf[lane], KERNEL_EPS, KERNEL_ONE_MINUS_EPS);
                quotes[i + lane].ask_p = kernel_clamp(ask_buf[lane], KERNEL_EPS, KERNEL_ONE_MINUS_EPS);
            }
        }
    }
#endif

    KERNEL_UNROLL_4
    for (; i < n; ++i) {
        const double gamma = fmax(0.0, states[i].gamma);
        const double tau = fmax(0.0, states[i].tau);
        const double k = fmax(KERNEL_EPS, states[i].k);

        const double sigma2 = states[i].sigma_b * states[i].sigma_b;
        const double variance_horizon = sigma2 * tau;
        const double risk_term = gamma * variance_horizon;

        const double r_x = states[i].x_t - (states[i].q_t * risk_term);
        const double delta_x =
            (0.5 * risk_term) + (log1p(gamma / k) / k);

        const double bid_x = r_x - delta_x;
        const double ask_x = r_x + delta_x;

        quotes[i].bid_p = kernel_sigmoid_internal(bid_x);
        quotes[i].ask_p = kernel_sigmoid_internal(ask_x);
    }
}
