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

/*
 * Fast AVX-512 log1p approximation for x >= 0:
 * log1p(x) = log(1 + x) = log(m * 2^e) = log(m) + e*ln(2), m in [1, 2).
 * log(m) uses atanh transform with odd-power Horner polynomial:
 * log(m) = 2*z*(1 + z^2/3 + z^4/5 + ... + z^12/13), z=(m-1)/(m+1).
 */
static inline __m512d kernel_log1p_fast_avx512(__m512d x) {
    const __m512d v_zero = _mm512_set1_pd(0.0);
    const __m512d v_one = _mm512_set1_pd(1.0);
    const __m512d v_two = _mm512_set1_pd(2.0);
    const __m512d v_ln2 = _mm512_set1_pd(0.693147180559945309417232121458176568);

    const __m512i v_exp_mask = _mm512_set1_epi64(0x7ffULL);
    const __m512i v_exp_bias = _mm512_set1_epi64(1023);
    const __m512i v_mant_mask = _mm512_set1_epi64(0x000fffffffffffffULL);
    const __m512i v_one_exp_bits = _mm512_set1_epi64(0x3ff0000000000000ULL);

    const __m512d x_nonneg = _mm512_max_pd(v_zero, x);
    const __m512d y = _mm512_add_pd(v_one, x_nonneg);
    const __m512i y_bits = _mm512_castpd_si512(y);

    const __m512i exp_bits = _mm512_and_si512(_mm512_srli_epi64(y_bits, 52), v_exp_mask);
    const __m512i mant_bits = _mm512_and_si512(y_bits, v_mant_mask);
    const __m512i norm_bits = _mm512_or_si512(mant_bits, v_one_exp_bits);

    const __m512d m = _mm512_castsi512_pd(norm_bits);
    const __m512d z = _mm512_div_pd(_mm512_sub_pd(m, v_one), _mm512_add_pd(m, v_one));
    const __m512d z2 = _mm512_mul_pd(z, z);

    __m512d poly = _mm512_set1_pd(1.0 / 13.0);
    poly = _mm512_add_pd(_mm512_set1_pd(1.0 / 11.0), _mm512_mul_pd(z2, poly));
    poly = _mm512_add_pd(_mm512_set1_pd(1.0 / 9.0), _mm512_mul_pd(z2, poly));
    poly = _mm512_add_pd(_mm512_set1_pd(1.0 / 7.0), _mm512_mul_pd(z2, poly));
    poly = _mm512_add_pd(_mm512_set1_pd(1.0 / 5.0), _mm512_mul_pd(z2, poly));
    poly = _mm512_add_pd(_mm512_set1_pd(1.0 / 3.0), _mm512_mul_pd(z2, poly));
    poly = _mm512_add_pd(v_one, _mm512_mul_pd(z2, poly));

    const __m512d ln_m = _mm512_mul_pd(v_two, _mm512_mul_pd(z, poly));

    const __m512i exp_unbiased_64 = _mm512_sub_epi64(exp_bits, v_exp_bias);
    const __m256i exp_unbiased_32 = _mm512_cvtepi64_epi32(exp_unbiased_64);
    const __m512d exp_unbiased = _mm512_cvtepi32_pd(exp_unbiased_32);

    __m512d ln_y = _mm512_add_pd(_mm512_mul_pd(exp_unbiased, v_ln2), ln_m);

    const __mmask8 special_mask = _mm512_cmp_epi64_mask(exp_bits, v_exp_mask, _MM_CMPINT_EQ);
    ln_y = _mm512_mask_mov_pd(ln_y, special_mask, x);

    return ln_y;
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
    const double *x_t,
    const double *q_t,
    const double *sigma_b,
    const double *gamma,
    const double *tau,
    const double *k,
    double *bid_p,
    double *ask_p,
    size_t n
) {
    if (x_t == NULL || q_t == NULL || sigma_b == NULL || gamma == NULL || tau == NULL || k == NULL ||
        bid_p == NULL || ask_p == NULL || n == 0u) {
        return;
    }

    size_t i = 0u;

#if defined(__AVX512F__)
    {
        const __m512d v_zero = _mm512_set1_pd(0.0);
        const __m512d v_half = _mm512_set1_pd(0.5);
        const __m512d v_eps = _mm512_set1_pd(KERNEL_EPS);

        for (; (i + 7u) < n; i += 8u) {
            const __m512d x = _mm512_loadu_pd(x_t + i);
            const __m512d q = _mm512_loadu_pd(q_t + i);
            const __m512d sigma = _mm512_loadu_pd(sigma_b + i);
            const __m512d gamma_raw = _mm512_loadu_pd(gamma + i);
            const __m512d tau_raw = _mm512_loadu_pd(tau + i);
            const __m512d k_raw = _mm512_loadu_pd(k + i);

            const __m512d gamma_v = _mm512_max_pd(v_zero, gamma_raw);
            const __m512d tau_v = _mm512_max_pd(v_zero, tau_raw);
            const __m512d k_v = _mm512_max_pd(v_eps, k_raw);

            const __m512d sigma2 = _mm512_mul_pd(sigma, sigma);
            const __m512d variance_horizon = _mm512_mul_pd(sigma2, tau_v);
            const __m512d risk_term = _mm512_mul_pd(gamma_v, variance_horizon);
            const __m512d r_x = _mm512_sub_pd(x, _mm512_mul_pd(q, risk_term));

            const __m512d gamma_over_k = _mm512_div_pd(gamma_v, k_v);
            const __m512d linear_half_spread = _mm512_mul_pd(v_half, risk_term);
            const __m512d non_linear = _mm512_div_pd(kernel_log1p_fast_avx512(gamma_over_k), k_v);
            const __m512d delta_x = _mm512_add_pd(linear_half_spread, non_linear);

            const __m512d x_bid = _mm512_sub_pd(r_x, delta_x);
            const __m512d x_ask = _mm512_add_pd(r_x, delta_x);

            const __m512d bid = kernel_sigmoid_fast_avx512(x_bid);
            const __m512d ask = kernel_sigmoid_fast_avx512(x_ask);

            _mm512_storeu_pd(bid_p + i, bid);
            _mm512_storeu_pd(ask_p + i, ask);
        }
    }
#endif

    KERNEL_UNROLL_4
    for (; i < n; ++i) {
        const double gamma_v = fmax(0.0, gamma[i]);
        const double tau_v = fmax(0.0, tau[i]);
        const double k_v = fmax(KERNEL_EPS, k[i]);

        const double sigma2 = sigma_b[i] * sigma_b[i];
        const double variance_horizon = sigma2 * tau_v;
        const double risk_term = gamma_v * variance_horizon;
        const double r_x = x_t[i] - (q_t[i] * risk_term);

        const double delta_x = (0.5 * risk_term) + (log1p(gamma_v / k_v) / k_v);
        const double bid_x = r_x - delta_x;
        const double ask_x = r_x + delta_x;

        bid_p[i] = kernel_sigmoid_internal(bid_x);
        ask_p[i] = kernel_sigmoid_internal(ask_x);
    }
}
