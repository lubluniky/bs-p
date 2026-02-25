use core::ffi::c_int;
use core::marker::PhantomData;
use core::mem::MaybeUninit;
use core::ptr::NonNull;

const CACHELINE: usize = 64;
const INDEX_PAD: usize = CACHELINE - core::mem::size_of::<u64>();

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct L2Update {
    pub market_id: u64,
    pub mid_price: f64,
    pub implied_vol: f64,
}

#[repr(C, align(64))]
pub struct SpscRingBuffer {
    _head: u64,
    _head_pad: [u8; INDEX_PAD],
    _tail: u64,
    _tail_pad: [u8; INDEX_PAD],
    _slots: *mut L2Update,
    _capacity: u64,
    _mask: u64,
}

impl Default for SpscRingBuffer {
    fn default() -> Self {
        Self {
            _head: 0,
            _head_pad: [0; INDEX_PAD],
            _tail: 0,
            _tail_pad: [0; INDEX_PAD],
            _slots: core::ptr::null_mut(),
            _capacity: 0,
            _mask: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RingInitError {
    InvalidCapacity,
    InitFailed,
}

unsafe extern "C" {
    fn spsc_ring_buffer_init(rb: *mut SpscRingBuffer, slots: *mut L2Update, capacity: u64) -> c_int;
    fn spsc_ring_buffer_push(rb: *mut SpscRingBuffer, msg: *const L2Update) -> c_int;
    fn spsc_ring_buffer_pop(rb: *mut SpscRingBuffer, out: *mut L2Update) -> c_int;
    fn spsc_ring_buffer_capacity(rb: *const SpscRingBuffer) -> u64;
    fn spsc_ring_buffer_len(rb: *const SpscRingBuffer) -> u64;
    fn spsc_ring_buffer_is_empty(rb: *const SpscRingBuffer) -> c_int;
    fn spsc_ring_buffer_is_full(rb: *const SpscRingBuffer) -> c_int;
}

pub struct Producer<'a> {
    ring: NonNull<SpscRingBuffer>,
    _marker: PhantomData<(&'a SpscRingBuffer, &'a [L2Update])>,
}

pub struct Consumer<'a> {
    ring: NonNull<SpscRingBuffer>,
    _marker: PhantomData<(&'a SpscRingBuffer, &'a [L2Update])>,
}

pub fn split<'a>(
    ring: &'a mut SpscRingBuffer,
    slots: &'a mut [L2Update],
) -> Result<(Producer<'a>, Consumer<'a>), RingInitError> {
    if slots.len() < 2 || !slots.len().is_power_of_two() {
        return Err(RingInitError::InvalidCapacity);
    }

    let rc = unsafe { spsc_ring_buffer_init(ring as *mut _, slots.as_mut_ptr(), slots.len() as u64) };
    if rc != 1 {
        return Err(RingInitError::InitFailed);
    }

    let ptr = NonNull::from(ring);
    Ok((
        Producer {
            ring: ptr,
            _marker: PhantomData,
        },
        Consumer {
            ring: ptr,
            _marker: PhantomData,
        },
    ))
}

impl Producer<'_> {
    #[inline]
    pub fn try_push(&mut self, update: L2Update) -> Result<(), L2Update> {
        let rc = unsafe { spsc_ring_buffer_push(self.ring.as_ptr(), &update as *const _) };
        if rc == 1 {
            Ok(())
        } else {
            Err(update)
        }
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        unsafe { spsc_ring_buffer_capacity(self.ring.as_ptr()) as usize }
    }

    #[inline]
    pub fn len(&self) -> usize {
        unsafe { spsc_ring_buffer_len(self.ring.as_ptr()) as usize }
    }

    #[inline]
    pub fn is_full(&self) -> bool {
        unsafe { spsc_ring_buffer_is_full(self.ring.as_ptr()) == 1 }
    }
}

impl Consumer<'_> {
    #[inline]
    pub fn try_pop(&mut self) -> Option<L2Update> {
        let mut out = MaybeUninit::<L2Update>::uninit();
        let rc = unsafe { spsc_ring_buffer_pop(self.ring.as_ptr(), out.as_mut_ptr()) };
        if rc == 1 {
            Some(unsafe { out.assume_init() })
        } else {
            None
        }
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        unsafe { spsc_ring_buffer_capacity(self.ring.as_ptr()) as usize }
    }

    #[inline]
    pub fn len(&self) -> usize {
        unsafe { spsc_ring_buffer_len(self.ring.as_ptr()) as usize }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        unsafe { spsc_ring_buffer_is_empty(self.ring.as_ptr()) == 1 }
    }
}

// SAFETY: each endpoint is intended for single-thread ownership and uses only
// lock-free atomic coordination in C. Moving endpoints across threads is safe.
unsafe impl Send for Producer<'_> {}
unsafe impl Send for Consumer<'_> {}

// SAFETY: sharing references is safe because mutation APIs require `&mut self`
// and all cross-thread synchronization is handled by the underlying atomics.
unsafe impl Sync for Producer<'_> {}
unsafe impl Sync for Consumer<'_> {}
