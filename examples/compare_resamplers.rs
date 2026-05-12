//! Resampler benchmark: runs the 2D vehicle filter with three
//! different resamplers and reports wall-clock time per step at
//! several values of N. Demonstrates the practical value of
//! ltsis's linear-time resampler over a naive
//! cumulative-sum + binary-search baseline.
//!
//! Resamplers compared:
//!
//! - `ltsis_buffered`: ltsis::sample_indices_buffered. The
//!   bit-stashed buffered variant; the default in `ltbpf`.
//! - `ltsis_streaming`: ltsis::sample_indices. Iterator-driven,
//!   same algorithm, no bit-stash. Slightly slower than buffered.
//! - `naive`: prefix-sum + per-draw binary search. O(n log n).
//!   The textbook resampler; what most particle-filter
//!   implementations ship with.
//!
//! CSV is written to stdout. Run with:
//!
//!     cargo run --release --example compare_resamplers > resamplers.csv

use std::time::{Duration, Instant};

use rand::rngs::SmallRng;
use rand::RngExt;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

// -- Model parameters (mirrors examples/vehicle.rs) -----------------

const DT: f32 = 0.1;
const SIGMA_A: f32 = 0.5;
const SIGMA_GPS: f32 = 5.0;
const SIGMA_IMU: f32 = 0.2;
const STEPS: usize = 200;

#[derive(Clone, Default)]
struct Vehicle {
    x: f32,
    y: f32,
    vx: f32,
    vy: f32,
}

struct Obs {
    gps_x: f32,
    gps_y: f32,
    imu_vx: f32,
    imu_vy: f32,
}

fn sample_prior(rng: &mut SmallRng) -> Vehicle {
    let pos = Normal::new(0.0_f32, 5.0).unwrap();
    let vel = Normal::new(0.0_f32, 2.0).unwrap();
    Vehicle {
        x: pos.sample(rng),
        y: pos.sample(rng),
        vx: vel.sample(rng),
        vy: vel.sample(rng),
    }
}

fn propagate(rng: &mut SmallRng, s: &Vehicle) -> Vehicle {
    let an = Normal::new(0.0_f32, SIGMA_A).unwrap();
    let ax = an.sample(rng);
    let ay = an.sample(rng);
    Vehicle {
        x: s.x + s.vx * DT + 0.5 * ax * DT * DT,
        y: s.y + s.vy * DT + 0.5 * ay * DT * DT,
        vx: s.vx + ax * DT,
        vy: s.vy + ay * DT,
    }
}

fn weight_update(p: &Vehicle, obs: &Obs) -> f32 {
    let r1 = (obs.gps_x - p.x) / SIGMA_GPS;
    let r2 = (obs.gps_y - p.y) / SIGMA_GPS;
    let r3 = (obs.imu_vx - p.vx) / SIGMA_IMU;
    let r4 = (obs.imu_vy - p.vy) / SIGMA_IMU;
    (-0.5 * (r1 * r1 + r2 * r2 + r3 * r3 + r4 * r4))
        .max(-50.0)
        .exp()
}

fn sense(rng: &mut SmallRng, truth: &Vehicle) -> Obs {
    let gps = Normal::new(0.0_f32, SIGMA_GPS).unwrap();
    let imu = Normal::new(0.0_f32, SIGMA_IMU).unwrap();
    Obs {
        gps_x: truth.x + gps.sample(rng),
        gps_y: truth.y + gps.sample(rng),
        imu_vx: truth.vx + imu.sample(rng),
        imu_vy: truth.vy + imu.sample(rng),
    }
}

// -- Naive resampler ------------------------------------------------

/// Prefix-sum + binary search. O(n) to build the prefix sums, O(n log n)
/// for the n binary searches. Writes `n` sampled indices into `out`
/// (not sorted; the gather step doesn't care).
///
/// `scratch` must have length >= weights.len(); it holds the prefix
/// sums. Avoids allocation.
fn naive_resample(rng: &mut SmallRng, weights: &[f32], scratch: &mut [f32], out: &mut [u32]) {
    let n = weights.len();
    assert_eq!(scratch.len(), n);
    let mut acc = 0.0_f32;
    for i in 0..n {
        acc += weights[i];
        scratch[i] = acc;
    }
    let total = acc;
    for slot in out.iter_mut() {
        let u: f32 = rng.random::<f32>() * total;
        // Binary search for the first cum >= u.
        let mut lo = 0usize;
        let mut hi = n;
        while lo < hi {
            let mid = (lo + hi) / 2;
            if scratch[mid] < u {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        *slot = lo.min(n - 1) as u32;
    }
}

// -- One inner loop, parameterized by resampler ---------------------

#[derive(Clone, Copy)]
enum Which {
    Buffered,
    Streaming,
    Naive,
}

fn bench(n: usize, which: Which) -> Duration {
    let mut rng = SmallRng::seed_from_u64(0xC0FFEE);
    let mut p_curr: Vec<Vehicle> = (0..n).map(|_| sample_prior(&mut rng)).collect();
    let mut p_next = vec![Vehicle::default(); n];
    let mut weights = vec![1.0_f32; n];
    let mut idx = vec![0_u32; n];
    let mut scratch = vec![0.0_f32; n];
    let mut truth = Vehicle {
        vx: 1.0,
        vy: 0.5,
        ..Vehicle::default()
    };

    let start = Instant::now();
    for _ in 0..STEPS {
        truth = propagate(&mut rng, &truth);
        let obs = sense(&mut rng, &truth);

        // Propagate.
        for i in 0..n {
            p_next[i] = propagate(&mut rng, &p_curr[i]);
        }
        // Weight + max.
        let mut max_w = 0.0_f32;
        for i in 0..n {
            weights[i] *= weight_update(&p_next[i], &obs);
            if weights[i] > max_w {
                max_w = weights[i];
            }
        }
        // Bail rather than fight underflow in the bench.
        if max_w == 0.0 {
            weights.fill(1.0);
            p_curr.clone_from_slice(&p_next);
            continue;
        }
        let inv = 1.0 / max_w;
        for w in &mut weights {
            *w *= inv;
        }

        // Resample (unconditional — keep the comparison apples-to-apples).
        match which {
            Which::Buffered => {
                ltsis::sample_indices_buffered(&mut rng, &weights, &mut idx);
                for i in 0..n {
                    p_curr[i] = p_next[idx[i] as usize].clone();
                }
            }
            Which::Streaming => {
                let it = ltsis::sample_indices(&mut rng, &weights, n as u32);
                for (i, j) in it.enumerate() {
                    p_curr[i] = p_next[j as usize].clone();
                }
            }
            Which::Naive => {
                naive_resample(&mut rng, &weights, &mut scratch, &mut idx);
                for i in 0..n {
                    p_curr[i] = p_next[idx[i] as usize].clone();
                }
            }
        }
        weights.fill(1.0);
    }
    start.elapsed()
}

fn main() {
    let sizes: &[usize] = &[100, 300, 1000, 3000, 10000, 30000];
    println!("n,resampler,total_ms,per_step_us");
    for &n in sizes {
        for which in [Which::Buffered, Which::Streaming, Which::Naive] {
            let label = match which {
                Which::Buffered => "ltsis_buffered",
                Which::Streaming => "ltsis_streaming",
                Which::Naive => "naive",
            };
            let elapsed = bench(n, which);
            let total_ms = elapsed.as_secs_f64() * 1000.0;
            let per_step_us = elapsed.as_secs_f64() * 1_000_000.0 / STEPS as f64;
            println!("{n},{label},{total_ms:.3},{per_step_us:.3}");
        }
    }
}
