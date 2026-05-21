//! Convergence test: on a 1D linear-Gaussian state-space model, a
//! Kalman filter gives the exact posterior. The BPF should track that
//! posterior as N grows; we verify by running both on the same
//! observation streams and asserting that the BPF weighted mean stays
//! close to the Kalman mean (in units of Kalman posterior standard
//! deviation, averaged over trials).
//!
//! Model:
//!     x_t = x_{t-1} + w_t,   w_t ~ N(0, Q)
//!     y_t = x_t       + v_t, v_t ~ N(0, R)
//!     x_0 ~ N(m0, P0)
//!
//! This is the regime where Kalman is optimal and BPF is, asymptotically
//! in N, equivalent. So this test is a sanity check on the filter
//! plumbing — not a demonstration that BPF beats Kalman (it doesn't
//! here).

use ltbpf::{weighted_mean, Buffers, Coord, ParticleFilter};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

const Q: f32 = 0.1; // process noise variance
const R: f32 = 1.0; // observation noise variance
const M0: f32 = 0.0;
const P0: f32 = 4.0;

const N_PARTICLES: usize = 500;
const STEPS: usize = 100;
const N_TRIALS: usize = 50;

/// One Kalman update on the 1D scalar model.
fn kalman_step(m: &mut f32, p: &mut f32, y: f32) {
    // Predict.
    *p += Q;
    // Update with y.
    let k = *p / (*p + R);
    *m += k * (y - *m);
    *p *= 1.0 - k;
}

/// Weighted variance estimator for f32 particles.
fn weighted_variance(particles: &[f32], weights: &[f32], mean: f32) -> f32 {
    let mut sw = 0.0_f32;
    let mut swsq = 0.0_f32;
    for (&x, &w) in particles.iter().zip(weights) {
        let d = x - mean;
        swsq += w * d * d;
        sw += w;
    }
    swsq / sw
}

#[test]
fn bpf_tracks_kalman_on_1d_linear_gaussian() {
    let sigma_q = Q.sqrt();
    let sigma_r = R.sqrt();
    let sigma_p0 = P0.sqrt();

    // Aggregates over (trial, step) pairs.
    let mut total_normalized_err = 0.0_f64;
    let mut total_count: u32 = 0;
    let mut max_normalized_err: f32 = 0.0;

    // Variance check — at the last step of each trial, ratio of BPF
    // weighted variance to Kalman P. Averaged across trials, this
    // should be close to 1.
    let mut var_ratio_sum = 0.0_f64;

    for trial in 0..N_TRIALS {
        let mut rng = SmallRng::seed_from_u64(0xABCD_0000 + trial as u64);

        let prior = Normal::new(M0, sigma_p0).unwrap();
        let proc_noise = Normal::new(0.0_f32, sigma_q).unwrap();
        let obs_noise = Normal::new(0.0_f32, sigma_r).unwrap();

        // Initial particles from the prior.
        let mut p_curr: Vec<f32> = (0..N_PARTICLES).map(|_| prior.sample(&mut rng)).collect();
        let mut p_next: Vec<f32> = vec![0.0; N_PARTICLES];
        let mut w = vec![1.0_f32; N_PARTICLES];
        let mut idx = vec![0_u32; N_PARTICLES];

        let mut filter = ParticleFilter::new(
            Buffers {
                particles_curr: &mut p_curr,
                particles_next: &mut p_next,
                weights: &mut w,
                indices: &mut idx,
            },
            |rng: &mut SmallRng, s: &f32| s + proc_noise.sample(rng),
            |s: &f32, y: &f32| {
                let r = (y - s) / sigma_r;
                (-0.5 * r * r).exp()
            },
        );

        // Kalman state.
        let mut km = M0;
        let mut kp = P0;
        // True latent state.
        let mut truth = prior.sample(&mut rng);

        for _step in 0..STEPS {
            // Advance truth and generate observation.
            truth += proc_noise.sample(&mut rng);
            let y = truth + obs_noise.sample(&mut rng);

            // Run filters.
            kalman_step(&mut km, &mut kp, y);
            filter.step(&mut rng, &y).expect("bpf step ok");

            // Compare BPF mean to Kalman mean in units of Kalman SD.
            let centroid = weighted_mean(filter.particles(), filter.weights(), |x| {
                [Coord::Linear(*x)]
            });
            let Coord::Linear(bpf_mean) = centroid[0] else {
                unreachable!()
            };
            let normalized = (bpf_mean - km).abs() / kp.sqrt();
            total_normalized_err += normalized as f64;
            total_count += 1;
            if normalized > max_normalized_err {
                max_normalized_err = normalized;
            }
        }

        // Variance comparison at end of trial.
        let centroid = weighted_mean(filter.particles(), filter.weights(), |x| {
            [Coord::Linear(*x)]
        });
        let Coord::Linear(bpf_mean) = centroid[0] else {
            unreachable!()
        };
        let bpf_var = weighted_variance(filter.particles(), filter.weights(), bpf_mean);
        var_ratio_sum += (bpf_var / kp) as f64;
    }

    let mean_normalized_err = (total_normalized_err / total_count as f64) as f32;
    let mean_var_ratio = (var_ratio_sum / N_TRIALS as f64) as f32;

    // Diagnostics (visible on failure).
    println!("trials={N_TRIALS} steps={STEPS} n={N_PARTICLES}");
    println!("mean |bpf-kalman| / sqrt(P_kalman) = {mean_normalized_err:.4}");
    println!("max  |bpf-kalman| / sqrt(P_kalman) = {max_normalized_err:.4}");
    println!("mean BPF_variance / Kalman_P       = {mean_var_ratio:.4}");

    // Plan threshold: BPF mean within 0.5 Kalman SDs averaged across
    // trials. With N=500, achieved value is typically ~0.05.
    assert!(
        mean_normalized_err < 0.5,
        "mean normalized error {mean_normalized_err} >= 0.5 Kalman SDs"
    );
    // Loose ratio bound — at finite N the BPF variance is biased
    // downward (resampling depletes the cloud).
    assert!(
        mean_var_ratio > 0.5 && mean_var_ratio < 1.5,
        "BPF/Kalman variance ratio {mean_var_ratio} outside [0.5, 1.5]"
    );
}
