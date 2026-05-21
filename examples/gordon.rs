//! Bootstrap particle filter on the Gordon–Salmond–Smith (1993) scalar
//! nonlinear benchmark — the standard testbed for particle filters.
//!
//! Model:
//!   x_t = x_{t-1}/2 + 25 x_{t-1}/(1+x_{t-1}²) + 8 cos(1.2(t-1)) + v_t
//!   y_t = x_t²/20 + w_t
//!   v_t ~ N(0, Q=10),  w_t ~ N(0, R=1),  x_0 ~ N(0, 10)
//!
//! Outputs CSV to stdout: step, truth_x, est_x, ess, abs_err.
//!
//! Usage:
//!     cargo run --release --example gordon [n] > out.csv
//! Defaults to n = 500 particles, 100 steps.

use ltbpf::{weighted_mean, Buffers, Coord, ParticleFilter, StepResult};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

// -- Model parameters ------------------------------------------------

const Q: f32 = 10.0;      // process noise variance
const R: f32 = 1.0;       // observation noise variance
const X0_VAR: f32 = 10.0; // prior variance

// -- State type ------------------------------------------------------

/// Particle state. The `t` field carries the time index so that the
/// time-varying cos term in the dynamics is available inside `propagate`.
#[derive(Clone, Debug)]
struct State {
    x: f32,
    t: u32,
}

impl Default for State {
    fn default() -> Self {
        Self { x: 0.0, t: 0 }
    }
}

// -- Model functions -------------------------------------------------

fn dynamics(x: f32, t: u32) -> f32 {
    x / 2.0 + 25.0 * x / (1.0 + x * x) + 8.0 * (1.2 * t as f32).cos()
}

fn propagate(rng: &mut SmallRng, s: &State) -> State {
    let noise = Normal::new(0.0_f32, Q.sqrt()).unwrap().sample(rng);
    State { x: dynamics(s.x, s.t) + noise, t: s.t + 1 }
}

fn weight_update(s: &State, &y: &f32) -> f32 {
    let y_hat = s.x * s.x / 20.0;
    let r = (y - y_hat) / R.sqrt();
    (-0.5 * r * r).max(-50.0).exp()
}

fn simulate_step(rng: &mut SmallRng, truth: &State) -> (State, f32) {
    let x_new = dynamics(truth.x, truth.t)
        + Normal::new(0.0_f32, Q.sqrt()).unwrap().sample(rng);
    let y = x_new * x_new / 20.0
        + Normal::new(0.0_f32, R.sqrt()).unwrap().sample(rng);
    (State { x: x_new, t: truth.t + 1 }, y)
}

// -- Main ------------------------------------------------------------

fn main() -> Result<(), ltbpf::StepError> {
    let n: usize = std::env::args().nth(1).map_or(500, |s| {
        s.parse().expect("argument must be a positive integer")
    });
    let steps = 100;
    let mut rng = SmallRng::seed_from_u64(0xDEAD_BEEF);

    let prior = Normal::new(0.0_f32, X0_VAR.sqrt()).unwrap();
    let mut p0: Vec<State> = (0..n)
        .map(|_| State { x: prior.sample(&mut rng), t: 0 })
        .collect();
    let mut p1 = vec![State::default(); n];
    let mut w = vec![1.0_f32; n];
    let mut idx = vec![0_u32; n];

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

    let mut truth = State { x: prior.sample(&mut rng), t: 0 };
    println!("step,truth_x,est_x,ess,abs_err");
    for _ in 0..steps {
        let (truth_new, y) = simulate_step(&mut rng, &truth);
        truth = truth_new;
        let StepResult { ess, .. } = filter.step(&mut rng, &y)?;
        let [Coord::Linear(est_x)] = weighted_mean(
            filter.particles(),
            filter.weights(),
            |s| [Coord::Linear(s.x)],
        ) else {
            unreachable!()
        };
        let err = (est_x - truth.x).abs();
        println!("{},{:.4},{:.4},{:.1},{:.4}", truth.t - 1, truth.x, est_x, ess, err);
    }
    Ok(())
}
