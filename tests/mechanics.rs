//! Filter mechanics: ESS computation, weight normalization, resample /
//! no-resample branches, SIS-between-resamples behavior, length
//! invariants. None of these tests touch the statistical correctness
//! of the filter — that's the Kalman convergence test's job; this
//! file only checks the bookkeeping.

use ltbpf::{Buffers, ParticleFilter, ResamplerKind, StepResult};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

/// Convenience: build a filter with given propagate / weight closures.
fn make_filter<'a, P, W>(
    n: usize,
    init: &'a mut Vec<f32>,
    scratch: &'a mut Vec<f32>,
    weights: &'a mut Vec<f32>,
    indices: &'a mut Vec<u32>,
    propagate: P,
    weigh: W,
) -> ParticleFilter<'a, f32, SmallRng, (), P, W>
where
    P: FnMut(&mut SmallRng, &f32) -> f32,
    W: FnMut(&f32, &()) -> f32,
{
    *init = (0..n).map(|i| i as f32).collect();
    *scratch = vec![0.0; n];
    *weights = vec![1.0; n];
    *indices = vec![0; n];
    ParticleFilter::new(
        Buffers {
            particles_curr: init,
            particles_next: scratch,
            weights,
            indices,
        },
        propagate,
        weigh,
        0.5,
    )
}

fn approx(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol
}

// -------------------------------------------------------------------
// ESS tabular checks
// -------------------------------------------------------------------

#[test]
fn ess_uniform_weights_equals_n() {
    let n = 8;
    let (mut a, mut b, mut w, mut idx) = (vec![], vec![], vec![], vec![]);
    let mut filter = make_filter(
        n,
        &mut a,
        &mut b,
        &mut w,
        &mut idx,
        |_rng, s| *s,
        |_s, _o| 1.0,
    );
    let mut rng = SmallRng::seed_from_u64(1);
    let sr = filter.step(&mut rng, &()).expect("step ok");
    assert!(
        approx(sr.ess, n as f32, 1e-5),
        "uniform weights: ess = {} expected {}",
        sr.ess,
        n
    );
}

#[test]
fn ess_single_positive_weight_equals_one() {
    let n = 5;
    let (mut a, mut b, mut w, mut idx) = (vec![], vec![], vec![], vec![]);
    // particle i has value i; weight_update returns 1.0 iff value == 0.0,
    // else 0.0. So only particle 0 survives.
    let mut filter = make_filter(
        n,
        &mut a,
        &mut b,
        &mut w,
        &mut idx,
        |_rng, s| *s,
        |s, _o| if *s == 0.0 { 1.0 } else { 0.0 },
    )
    .with_ess_threshold(0.0); // never resample, keep weights intact
    let mut rng = SmallRng::seed_from_u64(2);
    let sr = filter.step(&mut rng, &()).expect("step ok");
    assert!(
        approx(sr.ess, 1.0, 1e-5),
        "single nonzero weight: ess = {}",
        sr.ess
    );
    assert!(approx(sr.sum_w, 1.0, 1e-5));
    assert!(approx(sr.sum_w_squared, 1.0, 1e-5));
}

#[test]
fn ess_two_equal_weights_equals_two() {
    let n = 7;
    let (mut a, mut b, mut w, mut idx) = (vec![], vec![], vec![], vec![]);
    let mut filter = make_filter(
        n,
        &mut a,
        &mut b,
        &mut w,
        &mut idx,
        |_rng, s| *s,
        |s, _o| if *s == 0.0 || *s == 1.0 { 1.0 } else { 0.0 },
    )
    .with_ess_threshold(0.0);
    let mut rng = SmallRng::seed_from_u64(3);
    let sr = filter.step(&mut rng, &()).expect("step ok");
    assert!(
        approx(sr.ess, 2.0, 1e-5),
        "two equal weights: ess = {}",
        sr.ess
    );
}

// -------------------------------------------------------------------
// Normalization preserves ratios
// -------------------------------------------------------------------

#[test]
fn normalize_preserves_ratios() {
    let n = 4;
    let (mut a, mut b, mut w, mut idx) = (vec![], vec![], vec![], vec![]);
    // Particle i gets raw weight (i+1). So pre-normalize weights are
    // [1, 2, 3, 4]; post-normalize they should be [1/4, 2/4, 3/4, 1].
    let mut filter = make_filter(
        n,
        &mut a,
        &mut b,
        &mut w,
        &mut idx,
        |_rng, s| *s,
        |s, _o| *s + 1.0,
    )
    .with_ess_threshold(0.0); // suppress resample so we can read weights
    let mut rng = SmallRng::seed_from_u64(4);
    filter.step(&mut rng, &()).expect("step ok");
    let w = filter.weights();
    let expected = [0.25, 0.5, 0.75, 1.0];
    for i in 0..n {
        assert!(
            approx(w[i], expected[i], 1e-6),
            "weight[{i}] = {} expected {}",
            w[i],
            expected[i]
        );
    }
}

// -------------------------------------------------------------------
// Always-resample resets weights to 1.0
// -------------------------------------------------------------------

#[test]
fn always_resample_resets_weights_to_one() {
    let n = 16;
    let (mut a, mut b, mut w, mut idx) = (vec![], vec![], vec![], vec![]);
    let mut filter = make_filter(
        n,
        &mut a,
        &mut b,
        &mut w,
        &mut idx,
        |rng, s| *s + rng.random::<f32>(),
        |s, _o| (*s).abs() + 1.0, // nontrivial nonuniform weights
    )
    .with_ess_threshold(1.0); // resample every step
    let mut rng = SmallRng::seed_from_u64(5);
    let sr = filter.step(&mut rng, &()).expect("step ok");
    assert!(sr.resampled, "expected resample with threshold 1.0");
    for (i, &wi) in filter.weights().iter().enumerate() {
        assert_eq!(wi, 1.0, "weight[{i}] after resample = {wi}");
    }
}

// -------------------------------------------------------------------
// SIS between resamples: weights carry, particles change
// -------------------------------------------------------------------

#[test]
fn no_resample_carries_weights_and_propagates_particles() {
    let n = 8;
    let (mut a, mut b, mut w, mut idx) = (vec![], vec![], vec![], vec![]);
    let mut filter = make_filter(
        n,
        &mut a,
        &mut b,
        &mut w,
        &mut idx,
        |rng, s| *s + rng.random::<f32>(),
        |_s, _o| 1.0,
    )
    .with_ess_threshold(0.0); // never resample
    let mut rng = SmallRng::seed_from_u64(6);
    let initial: Vec<f32> = filter.particles().to_vec();
    for _ in 0..10 {
        let sr: StepResult = filter.step(&mut rng, &()).expect("step ok");
        assert!(!sr.resampled, "expected no resample");
    }
    // Weights stayed at 1.0 (each step multiplies by 1.0, max-normalize
    // is identity, no resample so no reset).
    for &wi in filter.weights() {
        assert!(approx(wi, 1.0, 1e-6), "weight should be 1.0, got {wi}");
    }
    // Particles drifted from their initial values.
    let any_changed = filter
        .particles()
        .iter()
        .zip(initial.iter())
        .any(|(p, i)| p != i);
    assert!(any_changed, "expected propagation to drift particles");
}

// -------------------------------------------------------------------
// Length invariants
// -------------------------------------------------------------------

#[test]
fn length_invariants_hold_across_steps() {
    let n = 32;
    let (mut a, mut b, mut w, mut idx) = (vec![], vec![], vec![], vec![]);
    let mut filter = make_filter(
        n,
        &mut a,
        &mut b,
        &mut w,
        &mut idx,
        |rng, s| *s + 0.1 * rng.random::<f32>(),
        |s, _o| (-0.5 * *s * *s).exp() + 1e-6,
    );
    let mut rng = SmallRng::seed_from_u64(7);
    for _ in 0..50 {
        let _ = filter.step(&mut rng, &()).expect("step ok");
        assert_eq!(filter.n(), n);
        assert_eq!(filter.particles().len(), n);
        assert_eq!(filter.weights().len(), n);
    }
}

// -------------------------------------------------------------------
// StepError surfaces correctly
// -------------------------------------------------------------------

#[test]
fn all_zero_weights_returns_error() {
    let n = 4;
    let (mut a, mut b, mut w, mut idx) = (vec![], vec![], vec![], vec![]);
    let mut filter = make_filter(
        n,
        &mut a,
        &mut b,
        &mut w,
        &mut idx,
        |_rng, s| *s,
        |_s, _o| 0.0,
    );
    let mut rng = SmallRng::seed_from_u64(8);
    match filter.step(&mut rng, &()) {
        Err(ltbpf::StepError::AllWeightsZero) => {}
        other => panic!("expected AllWeightsZero, got {other:?}"),
    }
}

#[test]
fn nonfinite_weight_returns_error() {
    let n = 4;
    let (mut a, mut b, mut w, mut idx) = (vec![], vec![], vec![], vec![]);
    let mut filter = make_filter(
        n,
        &mut a,
        &mut b,
        &mut w,
        &mut idx,
        |_rng, s| *s,
        |_s, _o| f32::NAN,
    );
    let mut rng = SmallRng::seed_from_u64(9);
    match filter.step(&mut rng, &()) {
        Err(ltbpf::StepError::NonFiniteWeight) => {}
        other => panic!("expected NonFiniteWeight, got {other:?}"),
    }
}

// -------------------------------------------------------------------
// Streaming resampler: same invariants as buffered
// -------------------------------------------------------------------

#[test]
fn streaming_resampler_resets_weights_to_one() {
    let n = 16;
    let (mut a, mut b, mut w, mut idx) = (vec![], vec![], vec![], vec![]);
    let mut filter = make_filter(
        n,
        &mut a,
        &mut b,
        &mut w,
        &mut idx,
        |rng, s| *s + rng.random::<f32>(),
        |s, _o| (*s).abs() + 1.0,
    )
    .with_resampler(ResamplerKind::Streaming)
    .with_ess_threshold(1.0);
    let mut rng = SmallRng::seed_from_u64(10);
    let sr = filter.step(&mut rng, &()).expect("step ok");
    assert!(sr.resampled);
    for (i, &wi) in filter.weights().iter().enumerate() {
        assert_eq!(wi, 1.0, "weight[{i}] = {wi}");
    }
}

#[test]
fn streaming_resampler_leaves_indices_untouched() {
    // Streaming's contract is that it doesn't write the indices
    // buffer. Pre-fill it with a sentinel and check the sentinel
    // survives a resample step.
    let n = 8;
    let mut a: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let mut b = vec![0.0_f32; n];
    let mut w = vec![1.0_f32; n];
    let mut idx: Vec<u32> = (0..n as u32).map(|i| 0xDEAD_0000 | i).collect();
    let sentinel = idx.clone();
    {
        let mut filter = ParticleFilter::new(
            Buffers {
                particles_curr: &mut a,
                particles_next: &mut b,
                weights: &mut w,
                indices: &mut idx,
            },
            |rng: &mut SmallRng, s: &f32| *s + rng.random::<f32>(),
            |s: &f32, _o: &()| (*s).abs() + 1.0,
            0.5,
        )
        .with_resampler(ResamplerKind::Streaming)
        .with_ess_threshold(1.0);
        let mut rng = SmallRng::seed_from_u64(11);
        filter.step(&mut rng, &()).expect("step ok");
    }
    assert_eq!(
        idx, sentinel,
        "streaming resampler wrote to the indices buffer"
    );
}
