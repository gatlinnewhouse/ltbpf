//! Regularized Particle Filter on the vehicle tracking benchmark.
//!
//! Identical dynamics to the SIS example, but uses `rpf_step` which
//! jitters each particle after every resample. This prevents sample
//! impoverishment — the collapse in diversity that repeated resampling
//! causes in low-process-noise settings.
//!
//! Jitter follows Silverman's optimal bandwidth: each dimension perturbed
//! by N(0, (h · σ̂_dim)²) where h = n^{-1/(d+4)}, d = 4, and σ̂_dim is
//! the weighted empirical standard deviation of that dimension in the
//! current particle cloud. This is recomputed each step so the kernel
//! adapts as the cloud concentrates.
//!
//! Note: RPF benefit is most visible under low process noise (risk of
//! diversity collapse after repeated resampling). With the moderate
//! SIGMA_A used here the bootstrap PF is already diverse enough, so
//! the improvement over SIR is marginal.
//!
//! Usage:
//!     cargo run --release --example rpf [n] > out.csv
//! Defaults to n = 1000 particles, 1000 steps.

use ltbpf::{weighted_mean, Buffers, Coord, ParticleFilter, StepResult};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

// -- Model parameters ------------------------------------------------

const DT: f32 = 0.1;
const SIGMA_A: f32 = 0.5;
const SIGMA_GPS: f32 = 5.0;
const SIGMA_IMU: f32 = 0.2;

// -- State and observation types ------------------------------------

#[derive(Clone, Default, Debug)]
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

// -- Model functions -------------------------------------------------

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
    let exponent = -0.5 * (r1 * r1 + r2 * r2 + r3 * r3 + r4 * r4);
    exponent.max(-50.0).exp()
}

fn sample_initial_truth() -> Vehicle {
    Vehicle { x: 0.0, y: 0.0, vx: 1.0, vy: 0.5 }
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

/// Weighted empirical standard deviation for each state dimension.
/// Returns (σ_x, σ_y, σ_vx, σ_vy).
fn empirical_stds(particles: &[Vehicle], weights: &[f32]) -> (f32, f32, f32, f32) {
    let sum_w: f32 = weights.iter().sum();
    let inv = 1.0 / sum_w;
    let mx: f32 = particles.iter().zip(weights).map(|(p, w)| p.x * w).sum::<f32>() * inv;
    let my: f32 = particles.iter().zip(weights).map(|(p, w)| p.y * w).sum::<f32>() * inv;
    let mvx: f32 = particles.iter().zip(weights).map(|(p, w)| p.vx * w).sum::<f32>() * inv;
    let mvy: f32 = particles.iter().zip(weights).map(|(p, w)| p.vy * w).sum::<f32>() * inv;
    let var_x: f32 =
        particles.iter().zip(weights).map(|(p, w)| w * (p.x - mx).powi(2)).sum::<f32>() * inv;
    let var_y: f32 =
        particles.iter().zip(weights).map(|(p, w)| w * (p.y - my).powi(2)).sum::<f32>() * inv;
    let var_vx: f32 =
        particles.iter().zip(weights).map(|(p, w)| w * (p.vx - mvx).powi(2)).sum::<f32>() * inv;
    let var_vy: f32 =
        particles.iter().zip(weights).map(|(p, w)| w * (p.vy - mvy).powi(2)).sum::<f32>() * inv;
    (var_x.sqrt(), var_y.sqrt(), var_vx.sqrt(), var_vy.sqrt())
}

// -- Main loop -------------------------------------------------------

fn main() -> Result<(), ltbpf::StepError> {
    let n: usize = std::env::args().nth(1).map_or(1000, |s| {
        s.parse().expect("argument must be a positive integer")
    });
    let steps = 1000;
    let mut truth_rng = SmallRng::seed_from_u64(0xC0FFEE);
    let mut filter_rng = SmallRng::seed_from_u64(0xDEAD_BEEF);

    // Silverman's optimal bandwidth: h = n^{-1/(d+4)}, d = 4.
    let h: f32 = (n as f32).powf(-1.0 / 8.0);

    let mut p0 = vec![Vehicle::default(); n];
    let mut p1 = vec![Vehicle::default(); n];
    let mut w = vec![1.0_f32; n];
    let mut idx = vec![0_u32; n];

    for p in &mut p0 {
        *p = sample_prior(&mut filter_rng);
    }

    // Default threshold (0.5) — jitter fires only when the bootstrap PF
    // would resample, i.e. when ESS < 0.5·n. This restores diversity on
    // demand without adding cumulative noise on every step.
    let mut filter = ParticleFilter::new(
        Buffers {
            particles_curr: &mut p0,
            particles_next: &mut p1,
            weights: &mut w,
            indices: &mut idx,
        },
        propagate,
        weight_update,
    );

    let mut truth = sample_initial_truth();
    println!("step,truth_x,truth_y,est_x,est_y,ess,err");
    for step in 0..steps {
        truth = propagate(&mut truth_rng, &truth);
        let obs = sense(&mut truth_rng, &truth);

        // Compute per-dimension kernel bandwidth from the current cloud
        // before stepping (Silverman: h · σ̂_dim per dimension).
        let (sx, sy, svx, svy) =
            empirical_stds(filter.particles(), filter.weights());
        let bx = h * sx;
        let by = h * sy;
        let bvx = h * svx;
        let bvy = h * svy;

        let StepResult { ess, .. } = filter.rpf_step(&mut filter_rng, &obs, &mut |rng, s| {
            Vehicle {
                x: s.x + Normal::new(0.0_f32, bx.max(1e-6)).unwrap().sample(rng),
                y: s.y + Normal::new(0.0_f32, by.max(1e-6)).unwrap().sample(rng),
                vx: s.vx + Normal::new(0.0_f32, bvx.max(1e-6)).unwrap().sample(rng),
                vy: s.vy + Normal::new(0.0_f32, bvy.max(1e-6)).unwrap().sample(rng),
            }
        })?;

        let centroid = weighted_mean(filter.particles(), filter.weights(), |v| {
            [Coord::Linear(v.x), Coord::Linear(v.y)]
        });
        let (Coord::Linear(est_x), Coord::Linear(est_y)) = (centroid[0], centroid[1]) else {
            unreachable!()
        };

        let err = (est_x - truth.x).hypot(est_y - truth.y);
        println!("{step},{},{},{est_x},{est_y},{ess},{err}", truth.x, truth.y);
    }
    Ok(())
}
