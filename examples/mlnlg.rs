//! Mixed Linear/Nonlinear Gaussian (MLNLG) bootstrap particle filter.
//!
//! Extends the Gordon–Salmond–Smith scalar benchmark with a vector
//! linear channel state z driven by a block-diagonal rotation matrix A_z.
//! A Rao-Blackwellised filter (RBPF) could marginalise z analytically,
//! but this example uses a plain joint bootstrap filter since ltbpf does
//! not provide a Kalman update primitive.
//!
//! Model (L = 4 linear dimensions, two taps):
//!   ξ_t = ξ_{t-1}/2 + 25 ξ_{t-1}/(1+ξ_{t-1}²) + 8 cos(1.2(t-1)) + v_ξ
//!   z_t = A_z z_{t-1} + v_z
//!   y_t = C z_t + ξ_t²/20 + w
//!   v_ξ ~ N(0, Q_ξ=10),  v_z ~ N(0, Q_z=0.3·I),  w ~ N(0, R=1)
//!
//! A_z is block-diagonal with two 2×2 rotation-scaling blocks:
//!   tap 1: ρ=0.95, ω=0.05   tap 2: ρ=0.88, ω=0.15
//! C = [1, 1, 1, 1] (sums all z components).
//!
//! Outputs CSV to stdout.
//!
//! Usage:
//!     cargo run --release --example mlnlg [n] > out.csv
//! Defaults to n = 500 particles, 100 steps.

use ltbpf::{weighted_mean, Buffers, Coord, ParticleFilter, StepResult};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

// -- Model parameters ------------------------------------------------

const Q_XI: f32 = 10.0; // nonlinear state process noise variance
const QZ: f32 = 0.3; // linear state process noise variance (diagonal)
const R: f32 = 1.0; // observation noise variance
const XI0_VAR: f32 = 5.0; // prior variance for ξ
const Z0_VAR: f32 = 0.5; // prior variance per z component (Z0 = 0.5·I)

// -- A_z matrix helpers ----------------------------------------------

/// Build the 4×4 block-diagonal A_z from the two tap parameters.
fn make_az() -> [[f32; 4]; 4] {
    let (rho1, omega1) = (0.95_f32, 0.05_f32);
    let (rho2, omega2) = (0.88_f32, 0.15_f32);
    let (c1, s1) = (omega1.cos(), omega1.sin());
    let (c2, s2) = (omega2.cos(), omega2.sin());
    [
        [rho1 * c1, rho1 * s1, 0.0, 0.0],
        [-rho1 * s1, rho1 * c1, 0.0, 0.0],
        [0.0, 0.0, rho2 * c2, rho2 * s2],
        [0.0, 0.0, -rho2 * s2, rho2 * c2],
    ]
}

fn matvec4(m: &[[f32; 4]; 4], v: [f32; 4]) -> [f32; 4] {
    let mut out = [0.0_f32; 4];
    for i in 0..4 {
        for j in 0..4 {
            out[i] += m[i][j] * v[j];
        }
    }
    out
}

// -- State type ------------------------------------------------------

/// Joint particle state [ξ, z₀, z₁, z₂, z₃].
/// `t` carries the step index for the time-varying cos term.
#[derive(Clone, Debug)]
struct State {
    xi: f32,
    z: [f32; 4],
    t: u32,
}

impl Default for State {
    fn default() -> Self {
        Self {
            xi: 0.0,
            z: [0.0; 4],
            t: 0,
        }
    }
}

// -- Model functions -------------------------------------------------

fn xi_dynamics(xi: f32, t: u32) -> f32 {
    xi / 2.0 + 25.0 * xi / (1.0 + xi * xi) + 8.0 * (1.2 * t as f32).cos()
}

fn sample_z_noise(rng: &mut SmallRng) -> [f32; 4] {
    let d = Normal::new(0.0_f32, QZ.sqrt()).unwrap();
    [d.sample(rng), d.sample(rng), d.sample(rng), d.sample(rng)]
}

/// Returns the propagate closure, capturing `az` by value.
fn make_propagate(az: [[f32; 4]; 4]) -> impl FnMut(&mut SmallRng, &State) -> State {
    move |rng, s| {
        let xi_noise = Normal::new(0.0_f32, Q_XI.sqrt()).unwrap().sample(rng);
        let z_noise = sample_z_noise(rng);
        let z_pred = matvec4(&az, s.z);
        State {
            xi: xi_dynamics(s.xi, s.t) + xi_noise,
            z: [
                z_pred[0] + z_noise[0],
                z_pred[1] + z_noise[1],
                z_pred[2] + z_noise[2],
                z_pred[3] + z_noise[3],
            ],
            t: s.t + 1,
        }
    }
}

fn weight_update(s: &State, &y: &f32) -> f32 {
    // C = [1,1,1,1]: measurement is z_sum + ξ²/20
    let y_hat = s.z[0] + s.z[1] + s.z[2] + s.z[3] + s.xi * s.xi / 20.0;
    let r = (y - y_hat) / R.sqrt();
    (-0.5 * r * r).max(-50.0).exp()
}

fn simulate_step(rng: &mut SmallRng, truth: &State, az: &[[f32; 4]; 4]) -> (State, f32) {
    let xi_noise = Normal::new(0.0_f32, Q_XI.sqrt()).unwrap().sample(rng);
    let z_noise = sample_z_noise(rng);
    let xi_new = xi_dynamics(truth.xi, truth.t) + xi_noise;
    let z_pred = matvec4(az, truth.z);
    let z_new = [
        z_pred[0] + z_noise[0],
        z_pred[1] + z_noise[1],
        z_pred[2] + z_noise[2],
        z_pred[3] + z_noise[3],
    ];
    let obs_noise = Normal::new(0.0_f32, R.sqrt()).unwrap().sample(rng);
    let y = z_new[0] + z_new[1] + z_new[2] + z_new[3] + xi_new * xi_new / 20.0 + obs_noise;
    (
        State {
            xi: xi_new,
            z: z_new,
            t: truth.t + 1,
        },
        y,
    )
}

// -- Main ------------------------------------------------------------

fn main() -> Result<(), ltbpf::StepError> {
    let n: usize = std::env::args().nth(1).map_or(500, |s| {
        s.parse().expect("argument must be a positive integer")
    });
    let steps = 100;
    let mut rng = SmallRng::seed_from_u64(0xC0FFEE);

    let az = make_az();

    let prior_xi = Normal::new(0.0_f32, XI0_VAR.sqrt()).unwrap();
    let prior_z = Normal::new(0.0_f32, Z0_VAR.sqrt()).unwrap();

    let mut p0: Vec<State> = (0..n)
        .map(|_| State {
            xi: prior_xi.sample(&mut rng),
            z: [
                prior_z.sample(&mut rng),
                prior_z.sample(&mut rng),
                prior_z.sample(&mut rng),
                prior_z.sample(&mut rng),
            ],
            t: 0,
        })
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
        make_propagate(az),
        weight_update,
    );

    let mut truth = State {
        xi: prior_xi.sample(&mut rng),
        z: [
            prior_z.sample(&mut rng),
            prior_z.sample(&mut rng),
            prior_z.sample(&mut rng),
            prior_z.sample(&mut rng),
        ],
        t: 0,
    };

    println!(
        "step,truth_xi,est_xi,\
         truth_z0,est_z0,truth_z1,est_z1,\
         truth_z2,est_z2,truth_z3,est_z3,ess"
    );
    for _ in 0..steps {
        let (truth_new, y) = simulate_step(&mut rng, &truth, &az);
        truth = truth_new;
        let StepResult { ess, .. } = filter.step(&mut rng, &y)?;

        let est = weighted_mean(filter.particles(), filter.weights(), |s| {
            [
                Coord::Linear(s.xi),
                Coord::Linear(s.z[0]),
                Coord::Linear(s.z[1]),
                Coord::Linear(s.z[2]),
                Coord::Linear(s.z[3]),
            ]
        });
        let [est_xi, est_z0, est_z1, est_z2, est_z3] = est.map(|c| {
            let Coord::Linear(v) = c else { unreachable!() };
            v
        });

        println!(
            "{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.1}",
            truth.t - 1,
            truth.xi,
            est_xi,
            truth.z[0],
            est_z0,
            truth.z[1],
            est_z1,
            truth.z[2],
            est_z2,
            truth.z[3],
            est_z3,
            ess,
        );
    }
    Ok(())
}
