use std::hint::{black_box, spin_loop};
use std::thread;
use std::time::Instant;

use polymarket_kernel::calculate_quotes_logit;
use polymarket_kernel::ring_buffer::{L2Update, SpscRingBuffer, split};

const MARKETS: usize = 8_192;
const QUOTE_ITERS: usize = 100_000;

const SPSC_CAPACITY: usize = 1 << 20; // 1,048,576
const SPSC_MESSAGES: u64 = 10_000_000;

fn bench_quote_latency() -> f64 {
    let mut x_t = vec![0.0_f64; MARKETS];
    let mut q_t = vec![0.0_f64; MARKETS];
    let mut sigma_b = vec![0.0_f64; MARKETS];
    let mut gamma = vec![0.0_f64; MARKETS];
    let mut tau = vec![0.0_f64; MARKETS];
    let mut k = vec![0.0_f64; MARKETS];

    let mut bid_p = vec![0.0_f64; MARKETS];
    let mut ask_p = vec![0.0_f64; MARKETS];

    for i in 0..MARKETS {
        let fi = i as f64;
        x_t[i] = ((fi % 1000.0) / 1000.0 - 0.5) * 4.0; // [-2, 2]
        q_t[i] = (i as i64 % 41 - 20) as f64; // inventory in [-20, 20]
        sigma_b[i] = 0.10 + ((i % 200) as f64) * 0.0015; // [0.10, 0.3985]
        gamma[i] = 0.03 + ((i % 7) as f64) * 0.01; // [0.03, 0.09]
        tau[i] = 0.05 + ((i % 365) as f64) / 365.0; // [0.05, ~1.05]
        k[i] = 1.0 + ((i % 50) as f64) * 0.08; // [1.0, 4.92]
    }

    for _ in 0..2_000 {
        calculate_quotes_logit(
            &x_t, &q_t, &sigma_b, &gamma, &tau, &k, &mut bid_p, &mut ask_p,
        );
    }

    let start = Instant::now();
    for _ in 0..QUOTE_ITERS {
        calculate_quotes_logit(
            black_box(&x_t),
            black_box(&q_t),
            black_box(&sigma_b),
            black_box(&gamma),
            black_box(&tau),
            black_box(&k),
            black_box(&mut bid_p),
            black_box(&mut ask_p),
        );
    }
    let elapsed = start.elapsed();

    let total_updates = (MARKETS as u128) * (QUOTE_ITERS as u128);
    let ns_per_market = elapsed.as_nanos() as f64 / total_updates as f64;

    black_box((&bid_p, &ask_p));
    ns_per_market
}

fn bench_spsc_throughput() -> f64 {
    let mut ring = SpscRingBuffer::default();
    let mut slots = vec![L2Update::default(); SPSC_CAPACITY];

    let (mut producer, mut consumer) =
        split(&mut ring, &mut slots).expect("failed to initialize SPSC ring buffer");

    let start = Instant::now();

    thread::scope(|scope| {
        let producer_handle = scope.spawn(move || {
            for i in 0..SPSC_MESSAGES {
                let mut msg = L2Update {
                    market_id: i & 0x1fff,
                    mid_price: 0.45 + ((i % 10_000) as f64) * 0.00001,
                    implied_vol: 0.08 + ((i % 1_000) as f64) * 0.0001,
                };

                loop {
                    match producer.try_push(msg) {
                        Ok(()) => break,
                        Err(m) => {
                            msg = m;
                            spin_loop();
                        }
                    }
                }
            }
        });

        let consumer_handle = scope.spawn(move || {
            let mut received = 0_u64;
            let mut checksum = 0.0_f64;

            while received < SPSC_MESSAGES {
                if let Some(msg) = consumer.try_pop() {
                    received += 1;
                    checksum += msg.mid_price + msg.implied_vol + (msg.market_id as f64) * 1e-12;
                } else {
                    spin_loop();
                }
            }

            black_box(checksum);
            received
        });

        producer_handle
            .join()
            .expect("producer thread panicked during benchmark");

        let received = consumer_handle
            .join()
            .expect("consumer thread panicked during benchmark");
        assert_eq!(received, SPSC_MESSAGES, "message loss in SPSC benchmark");
    });

    let elapsed = start.elapsed();
    let secs = elapsed.as_secs_f64();
    (SPSC_MESSAGES as f64 / 1_000_000.0) / secs
}

fn main() {
    println!("============================================================");
    println!(" POLYMARKET-KERNEL RAW BENCHMARK ");
    println!("============================================================");
    println!(" Quote Batch Size        : {:>10} markets", MARKETS);
    println!(" Quote Iterations        : {:>10}", QUOTE_ITERS);

    let ns_per_market = bench_quote_latency();
    println!(
        " AVX-512 Quote Latency   : {:>10.2} ns/market",
        ns_per_market
    );

    println!("------------------------------------------------------------");
    println!(" SPSC Ring Capacity      : {:>10}", SPSC_CAPACITY);
    println!(" SPSC Messages           : {:>10}", SPSC_MESSAGES);

    let mops = bench_spsc_throughput();
    println!(" SPSC Throughput         : {:>10.2} M msgs/sec", mops);
    println!("============================================================");
}
