#include "ring_buffer.h"

#include <stddef.h>

static inline int spsc_is_power_of_two_u64(uint64_t v) {
    return (v != 0u) && ((v & (v - 1u)) == 0u);
}

static_assert(sizeof(_Atomic uint64_t) <= SPSC_CACHELINE_SIZE,
              "atomic index must fit in one cache line");
static_assert(offsetof(spsc_ring_buffer_t, head) % SPSC_CACHELINE_SIZE == 0u,
              "head must be cacheline aligned");
static_assert(offsetof(spsc_ring_buffer_t, tail) % SPSC_CACHELINE_SIZE == 0u,
              "tail must be cacheline aligned");

int spsc_ring_buffer_init(spsc_ring_buffer_t *rb, l2_update_t *slots, uint64_t capacity) {
    if (rb == NULL || slots == NULL) {
        return 0;
    }
    if (capacity < 2u || !spsc_is_power_of_two_u64(capacity)) {
        return 0;
    }

    atomic_init(&rb->head, 0u);
    atomic_init(&rb->tail, 0u);
    rb->slots = slots;
    rb->capacity = capacity;
    rb->mask = capacity - 1u;

    return 1;
}

int spsc_ring_buffer_push(spsc_ring_buffer_t *rb, const l2_update_t *msg) {
    if (rb == NULL || msg == NULL) {
        return 0;
    }

    const uint64_t head = atomic_load_explicit(&rb->head, memory_order_acquire);
    const uint64_t tail = atomic_load_explicit(&rb->tail, memory_order_acquire);

    if ((head - tail) >= rb->capacity) {
        return 0;
    }

    rb->slots[head & rb->mask] = *msg;
    atomic_store_explicit(&rb->head, head + 1u, memory_order_release);

    return 1;
}

int spsc_ring_buffer_pop(spsc_ring_buffer_t *rb, l2_update_t *out) {
    if (rb == NULL || out == NULL) {
        return 0;
    }

    const uint64_t tail = atomic_load_explicit(&rb->tail, memory_order_acquire);
    const uint64_t head = atomic_load_explicit(&rb->head, memory_order_acquire);

    if (tail == head) {
        return 0;
    }

    *out = rb->slots[tail & rb->mask];
    atomic_store_explicit(&rb->tail, tail + 1u, memory_order_release);

    return 1;
}

uint64_t spsc_ring_buffer_capacity(const spsc_ring_buffer_t *rb) {
    if (rb == NULL) {
        return 0u;
    }
    return rb->capacity;
}

uint64_t spsc_ring_buffer_len(const spsc_ring_buffer_t *rb) {
    if (rb == NULL) {
        return 0u;
    }

    const uint64_t head = atomic_load_explicit(&rb->head, memory_order_acquire);
    const uint64_t tail = atomic_load_explicit(&rb->tail, memory_order_acquire);
    return head - tail;
}

int spsc_ring_buffer_is_empty(const spsc_ring_buffer_t *rb) {
    return spsc_ring_buffer_len(rb) == 0u;
}

int spsc_ring_buffer_is_full(const spsc_ring_buffer_t *rb) {
    if (rb == NULL) {
        return 0;
    }
    return spsc_ring_buffer_len(rb) >= rb->capacity;
}
