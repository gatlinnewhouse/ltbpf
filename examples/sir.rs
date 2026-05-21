//! 2D vehicle tracking demo — reproduces Figure 4 of Massey (ICASSP
//! 2008). Near-constant-velocity dynamics, noisy GPS (position) + IMU
//! (velocity), Gaussian process noise. CSV to stdout.
//!
//! Usage:
//!     cargo run --release --example vehicle [n] > out.csv
//! Defaults to n = 1000 particles, 1000 steps.

use ltbpf::{weighted_mean, Buffers, Coord, ParticleFilter, StepResult};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

// -- Model parameters ------------------------------------------------

const DT: f32 = 0.1;
const SIGMA_A: f32 = 0.5; // process noise on acceleration
const SIGMA_GPS: f32 = 5.0; // GPS position noise
const SIGMA_IMU: f32 = 0.2; // IMU velocity noise

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

/// Product of four Gaussian likelihoods, evaluated in linear space.
/// Exponent is clamped at -50 so a momentarily-bad particle doesn't
/// underflow the entire cloud to zero — a demo-level mitigation, not
/// a library feature (see `ltbpf-plan.md` discussion).
fn weight_update(p: &Vehicle, obs: &Obs) -> f32 {
    let r1 = (obs.gps_x - p.x) / SIGMA_GPS;
    let r2 = (obs.gps_y - p.y) / SIGMA_GPS;
    let r3 = (obs.imu_vx - p.vx) / SIGMA_IMU;
    let r4 = (obs.imu_vy - p.vy) / SIGMA_IMU;
    let exponent = -0.5 * (r1 * r1 + r2 * r2 + r3 * r3 + r4 * r4);
    exponent.max(-50.0).exp()
}

fn sample_initial_truth() -> Vehicle {
    Vehicle {
        x: 0.0,
        y: 0.0,
        vx: 1.0,
        vy: 0.5,
    }
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

// -- Main loop -------------------------------------------------------

fn main() -> Result<(), ltbpf::StepError> {
    let n: usize = std::env::args().nth(1).map_or(1000, |s| {
        s.parse().expect("argument must be a positive integer")
    });
    let steps = 1000;
    let mut truth_rng = SmallRng::seed_from_u64(0xC0FFEE);
    let mut filter_rng = SmallRng::seed_from_u64(0xDEAD_BEEF);

    let mut p0 = vec![Vehicle::default(); n];
    let mut p1 = vec![Vehicle::default(); n];
    let mut w = vec![1.0_f32; n];
    let mut idx = vec![0_u32; n];

    for p in &mut p0 {
        *p = sample_prior(&mut filter_rng);
    }

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
        let StepResult { ess, .. } = filter.step(&mut filter_rng, &obs)?;

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
