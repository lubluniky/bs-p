#ifndef PM_SPSC_RING_BUFFER_H
#define PM_SPSC_RING_BUFFER_H

#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum { SPSC_CACHELINE_SIZE = 64 };

typedef struct {
    uint64_t market_id;
    double mid_price;
    double implied_vol;
} l2_update_t;

typedef struct {
    _Alignas(SPSC_CACHELINE_SIZE) _Atomic uint64_t head;
    uint8_t head_pad[SPSC_CACHELINE_SIZE - sizeof(_Atomic uint64_t)];

    _Alignas(SPSC_CACHELINE_SIZE) _Atomic uint64_t tail;
    uint8_t tail_pad[SPSC_CACHELINE_SIZE - sizeof(_Atomic uint64_t)];

    l2_update_t *slots;
    uint64_t capacity;
    uint64_t mask;
} spsc_ring_buffer_t;

int spsc_ring_buffer_init(spsc_ring_buffer_t *rb, l2_update_t *slots, uint64_t capacity);
int spsc_ring_buffer_push(spsc_ring_buffer_t *rb, const l2_update_t *msg);
int spsc_ring_buffer_pop(spsc_ring_buffer_t *rb, l2_update_t *out);

uint64_t spsc_ring_buffer_capacity(const spsc_ring_buffer_t *rb);
uint64_t spsc_ring_buffer_len(const spsc_ring_buffer_t *rb);
int spsc_ring_buffer_is_empty(const spsc_ring_buffer_t *rb);
int spsc_ring_buffer_is_full(const spsc_ring_buffer_t *rb);

#ifdef __cplusplus
}
#endif

#endif
