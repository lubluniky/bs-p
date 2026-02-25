#ifndef UNIFIED_PM_KERNEL_H
#define UNIFIED_PM_KERNEL_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    _Alignas(64) double x_t;
    double q_t;
    double sigma_b;
    double gamma;
    double tau;
    double k;
    double reserved0;
    double reserved1;
} market_state_t;

typedef struct {
    double bid_p;
    double ask_p;
} quote_out_t;

typedef struct {
    double delta_x;
    double gamma_x;
} greek_out_t;

double kernel_sigmoid(double x);
double kernel_logit(double p);

void kernel_sigmoid_batch(const double *x, double *out_p, size_t n);
void kernel_logit_batch(const double *p, double *out_x, size_t n);

void kernel_greeks_from_logit(double x, double *delta_x, double *gamma_x);
void kernel_greeks_batch(const double *x, greek_out_t *out, size_t n);

void calculate_quotes_logit(
    const market_state_t *states,
    quote_out_t *quotes,
    size_t n
);

#ifdef __cplusplus
}
#endif

#endif
