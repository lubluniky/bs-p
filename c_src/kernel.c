#include "kernel.h"

#include <math.h>
#include <stdatomic.h>
#include <stdint.h>

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#define PM_KERNEL_HAS_AVX512_DISPATCH 1
#define PM_KERNEL_TARGET_AVX512 __attribute__((target("avx512f")))
#include <immintrin.h>
#else
#define PM_KERNEL_HAS_AVX512_DISPATCH 0
#define PM_KERNEL_TARGET_AVX512
#endif

#if defined(__GNUC__) || defined(__clang__)
#define KERNEL_UNROLL_4 _Pragma("GCC unroll 4")
#else
#define KERNEL_UNROLL_4
#endif

static const double KERNEL_EPS = 1e-12;
static const double KERNEL_ONE_MINUS_EPS = 1.0 - 1e-12;

static _Atomic int kernel_has_avx512f_cache = -1;

static inline double kernel_clamp(double v, double lo, double hi) {
    return fmin(hi, fmax(lo, v));
}

static inline double kernel_sigmoid_exact(double x) {
    const double e = exp(-fabs(x));
    const double inv = 1.0 / (1.0 + e);
    const double p = (x >= 0.0) ? inv : (e * inv);
    return kernel_clamp(p, KERNEL_EPS, KERNEL_ONE_MINUS_EPS);
}

static inline int kernel_runtime_has_avx512f(void) {
    int cached = atomic_load_explicit(&kernel_has_avx512f_cache, memory_order_relaxed);
    if (cached >= 0) {
        return cached;
    }

    int has_avx512f = 0;

#if PM_KERNEL_HAS_AVX512_DISPATCH
    __builtin_cpu_init();
    has_avx512f = __builtin_cpu_supports("avx512f") ? 1 : 0;
#endif

    atomic_store_explicit(&kernel_has_avx512f_cache, has_avx512f, memory_order_relaxed);
    return has_avx512f;
}

double kernel_sigmoid(double x) {
    return kernel_sigmoid_exact(x);
}

double kernel_logit(double p) {
    const double pc = kernel_clamp(p, KERNEL_EPS, KERNEL_ONE_MINUS_EPS);
    return log(pc / (1.0 - pc));
}

void kernel_sigmoid_batch(const double *x, double *out_p, size_t n) {
    if (x == NULL || out_p == NULL || n == 0u) {
        return;
    }

    KERNEL_UNROLL_4
    for (size_t i = 0u; i < n; ++i) {
        out_p[i] = kernel_sigmoid_exact(x[i]);
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

    const double p = kernel_sigmoid_exact(x);
    const double delta = p * (1.0 - p);
    const double gamma = delta * (1.0 - (2.0 * p));

    *delta_x = delta;
    *gamma_x = gamma;
}

void kernel_greeks_batch(const double *x, greek_out_t *out, size_t n) {
    if (x == NULL || out == NULL || n == 0u) {
        return;
    }

    KERNEL_UNROLL_4
    for (size_t i = 0u; i < n; ++i) {
        kernel_greeks_from_logit(x[i], &out[i].delta_x, &out[i].gamma_x);
    }
}

void pm_internal_calculate_quotes_logit_portable(
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

    KERNEL_UNROLL_4
    for (size_t i = 0u; i < n; ++i) {
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

        bid_p[i] = kernel_sigmoid_exact(bid_x);
        ask_p[i] = kernel_sigmoid_exact(ask_x);
    }
}

#if PM_KERNEL_HAS_AVX512_DISPATCH
PM_KERNEL_TARGET_AVX512
void pm_internal_calculate_quotes_logit_avx512(
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

    const __m512d v_zero = _mm512_set1_pd(0.0);
    const __m512d v_half = _mm512_set1_pd(0.5);
    const __m512d v_eps = _mm512_set1_pd(KERNEL_EPS);

    size_t i = 0u;
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

        double gamma_buf[8];
        double k_buf[8];
        double non_linear_buf[8];
        _mm512_storeu_pd(gamma_buf, gamma_v);
        _mm512_storeu_pd(k_buf, k_v);

        for (size_t lane = 0u; lane < 8u; ++lane) {
            non_linear_buf[lane] = log1p(gamma_buf[lane] / k_buf[lane]) / k_buf[lane];
        }

        const __m512d linear_half_spread = _mm512_mul_pd(v_half, risk_term);
        const __m512d non_linear = _mm512_loadu_pd(non_linear_buf);
        const __m512d delta_x = _mm512_add_pd(linear_half_spread, non_linear);

        const __m512d x_bid = _mm512_sub_pd(r_x, delta_x);
        const __m512d x_ask = _mm512_add_pd(r_x, delta_x);

        double bid_x_buf[8];
        double ask_x_buf[8];
        _mm512_storeu_pd(bid_x_buf, x_bid);
        _mm512_storeu_pd(ask_x_buf, x_ask);

        for (size_t lane = 0u; lane < 8u; ++lane) {
            bid_p[i + lane] = kernel_sigmoid_exact(bid_x_buf[lane]);
            ask_p[i + lane] = kernel_sigmoid_exact(ask_x_buf[lane]);
        }
    }

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

        bid_p[i] = kernel_sigmoid_exact(bid_x);
        ask_p[i] = kernel_sigmoid_exact(ask_x);
    }
}
#endif

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
    if (kernel_runtime_has_avx512f()) {
#if PM_KERNEL_HAS_AVX512_DISPATCH
        pm_internal_calculate_quotes_logit_avx512(x_t, q_t, sigma_b, gamma, tau, k, bid_p, ask_p, n);
        return;
#endif
    }

    pm_internal_calculate_quotes_logit_portable(x_t, q_t, sigma_b, gamma, tau, k, bid_p, ask_p, n);
}
