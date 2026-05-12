//! Linear-time Bayesian Particle Filter.
//!
//! `ltbpf` implements the standard Sequential Importance Resampling
//! (SIR) bootstrap particle filter (Gordon, Salmond & Smith 1993) with
//! adaptive resampling triggered by effective sample size (Liu & Chen
//! 1995). The O(n) multinomial resampling step is delegated to
//! [`ltsis`].
//!
//! The library is `no_std`-clean and allocation-free: callers own the
//! particle and weight buffers and pass them in via [`Buffers`].
//! Dynamics and observation likelihoods are user-supplied closures, so
//! the model is whatever the caller can compute (nonlinear,
//! discontinuous, non-Gaussian — anything that produces a state sample
//! and a non-negative likelihood).
//!
//! See `ltbpf-plan.md` for the full design rationale.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(any(feature = "std", feature = "libm")))]
compile_error!(
    "Enable exactly one of the `std` or `libm` features so ltsis can \
     find its transcendental math."
);

use core::marker::PhantomData;

use rand::Rng;

/// Caller-owned buffers required to run a [`ParticleFilter`].
///
/// All four slices must have the same length `n >= 1`. The filter
/// borrows them for its lifetime and never allocates.
///
/// `particles_curr` and `particles_next` swap roles internally each
/// step; after a step returns, `particles_curr` always holds the
/// current particle cloud.
///
/// `indices` doubles as both scratch space for
/// [`ltsis::sample_indices_buffered`] and the index array that drives
/// the gather step.
pub struct Buffers<'a, S> {
    pub particles_curr: &'a mut [S],
    pub particles_next: &'a mut [S],
    pub weights: &'a mut [f32],
    pub indices: &'a mut [u32],
}

/// Per-step diagnostic output.
///
/// Returned in the `Ok` arm of [`ParticleFilter::step`]. Callers that
/// don't need the diagnostics can discard it; callers tuning a filter
/// (or logging health metrics) get all the relevant scalars without a
/// second pass over the weights.
#[derive(Debug, Clone, Copy)]
pub struct StepResult {
    /// Maximum particle weight observed during this step, after the
    /// user's `weight_update` was multiplied in but before the
    /// normalize-by-max pass.
    ///
    /// An absolute-scale check: extremely small values mean every
    /// particle had near-zero likelihood for the current observation.
    pub max_weight: f32,

    /// Σ wᵢ after normalize-by-max. In `[1, n]`, since every weight
    /// post-normalization is in `[0, 1]` with at least one equal to 1.
    pub sum_w: f32,

    /// Σ wᵢ² after normalize-by-max. Together with `sum_w` defines
    /// `ess`.
    pub sum_w_squared: f32,

    /// Effective sample size: `(Σ wᵢ)² / Σ wᵢ²`. In `[1, n]`.
    pub ess: f32,

    /// Whether this step triggered a resample.
    pub resampled: bool,
}

/// Reasons [`ParticleFilter::step`] can fail.
///
/// On error, the filter state is left unchanged from the caller's
/// perspective: particles, weights, and the curr/next role assignment
/// are not guaranteed to be untouched (propagation may have already
/// written into `particles_next`, and `weights` may have been
/// multiplied), but the caller should treat the filter as having
/// failed for this step and decide whether to reinitialize.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum StepError {
    /// Every particle's post-update weight is zero. The cloud has lost
    /// sync with the observation stream: typically because the
    /// observation is incompatible with every hypothesis (sensor
    /// failure, model misspecification, or a real "kidnapped robot"
    /// event). Recovery: reinitialize from a prior, widen process
    /// noise, or signal upstream.
    AllWeightsZero,

    /// `weight_update` returned a NaN, infinity, or negative value for
    /// at least one particle. Almost always a bug in the user's
    /// likelihood. Surfaced as an error rather than a panic so the
    /// caller can log and recover instead of crashing.
    NonFiniteWeight,
}

/// SIR Bayesian particle filter parameterized by:
///
/// - `S`: particle state type. Must be `Clone` (one clone per surviving
///   particle on resample steps).
/// - `R`: RNG type, threaded through to the user's `propagate` closure
///   and to `ltsis`.
/// - `Obs`: observation type, passed to `weight_update`.
/// - `Prop`: `FnMut(&mut R, &S) -> S` — samples from the transition
///   kernel, including process noise.
/// - `Weigh`: `FnMut(&S, &Obs) -> f32` — returns the per-step
///   multiplicative importance weight (non-negative, finite).
pub struct ParticleFilter<'a, S, R, Obs, Prop, Weigh>
where
    R: Rng + ?Sized,
    Prop: FnMut(&mut R, &S) -> S,
    Weigh: FnMut(&S, &Obs) -> f32,
{
    particles_curr: &'a mut [S],
    particles_next: &'a mut [S],
    weights: &'a mut [f32],
    indices: &'a mut [u32],
    propagate: Prop,
    weight_update: Weigh,
    ess_threshold: f32,
    _phantom: PhantomData<fn(&mut R, &Obs)>,
}

impl<'a, S, R, Obs, Prop, Weigh> ParticleFilter<'a, S, R, Obs, Prop, Weigh>
where
    R: Rng + ?Sized,
    Prop: FnMut(&mut R, &S) -> S,
    Weigh: FnMut(&S, &Obs) -> f32,
    S: Clone,
{
    /// Construct a filter from caller-owned buffers and the two model
    /// closures.
    ///
    /// All four buffers in `buffers` must have the same nonzero
    /// length. The caller is responsible for filling `particles_curr`
    /// with samples from the prior before the first [`Self::step`]
    /// call; `weights` is initialized to all-ones here.
    ///
    /// # Panics
    ///
    /// Panics if any buffer length differs from `particles_curr.len()`
    /// or if `particles_curr` is empty. These are programmer errors,
    /// not runtime conditions, so they are panics rather than errors.
    pub fn new(buffers: Buffers<'a, S>, propagate: Prop, weight_update: Weigh) -> Self {
        let Buffers {
            particles_curr,
            particles_next,
            weights,
            indices,
        } = buffers;
        let n = particles_curr.len();
        assert!(n > 0, "particle buffers must be nonempty");
        assert_eq!(particles_next.len(), n, "particles_next length mismatch");
        assert_eq!(weights.len(), n, "weights length mismatch");
        assert_eq!(indices.len(), n, "indices length mismatch");
        assert!(n <= u32::MAX as usize, "n must fit in u32");
        for w in weights.iter_mut() {
            *w = 1.0;
        }
        Self {
            particles_curr,
            particles_next,
            weights,
            indices,
            propagate,
            weight_update,
            ess_threshold: 0.5,
            _phantom: PhantomData,
        }
    }

    /// Set the effective-sample-size threshold for adaptive
    /// resampling. The filter resamples whenever
    /// `ess < threshold * n`. Default is `0.5`.
    ///
    /// `threshold = 1.0` forces resampling every step (the original
    /// Gordon-Salmond-Smith bootstrap filter); `threshold = 0.0`
    /// disables resampling entirely (pure SIS — useful only for tests).
    ///
    /// # Panics
    ///
    /// Panics if `threshold` is not finite or is outside `[0, 1]`.
    pub fn with_ess_threshold(mut self, threshold: f32) -> Self {
        assert!(threshold.is_finite(), "ess_threshold must be finite");
        assert!(
            (0.0..=1.0).contains(&threshold),
            "ess_threshold must be in [0, 1]"
        );
        self.ess_threshold = threshold;
        self
    }

    /// Particle count `n`. Constant for the life of the filter.
    pub fn n(&self) -> usize {
        self.particles_curr.len()
    }

    /// The current particle cloud.
    pub fn particles(&self) -> &[S] {
        self.particles_curr
    }

    /// The current particle weights, parallel to [`Self::particles`].
    /// Always in `[0, 1]` after a successful step (normalize-by-max
    /// ensures the largest weight is exactly `1.0`).
    pub fn weights(&self) -> &[f32] {
        self.weights
    }

    /// Advance the filter by one observation.
    ///
    /// Executes the standard SIR step:
    ///
    /// 1. Propagate each particle via the user's transition kernel.
    /// 2. Multiply weights by the user's observation likelihood.
    /// 3. Normalize weights so the maximum is `1.0`.
    /// 4. Compute ESS.
    /// 5. If `ess < ess_threshold * n`, resample (via
    ///    [`ltsis::sample_indices_buffered`]) and reset weights to
    ///    `1.0`; otherwise swap curr/next so weights carry forward
    ///    (SIS between resamples).
    pub fn step(&mut self, rng: &mut R, obs: &Obs) -> Result<StepResult, StepError> {
        let n = self.particles_curr.len();

        // 1. Propagate.
        for i in 0..n {
            self.particles_next[i] = (self.propagate)(rng, &self.particles_curr[i]);
        }

        // 2. Weight, tracking the running max.
        let mut max_w = 0.0_f32;
        for i in 0..n {
            let mul = (self.weight_update)(&self.particles_next[i], obs);
            if !mul.is_finite() || mul < 0.0 {
                return Err(StepError::NonFiniteWeight);
            }
            self.weights[i] *= mul;
            if !self.weights[i].is_finite() {
                return Err(StepError::NonFiniteWeight);
            }
            if self.weights[i] > max_w {
                max_w = self.weights[i];
            }
        }
        let max_weight = max_w;
        if max_weight == 0.0 {
            return Err(StepError::AllWeightsZero);
        }

        // 3. Normalize by max, accumulating the two moments needed for
        //    ESS in the same pass.
        let inv = 1.0 / max_weight;
        let mut sum_w = 0.0_f32;
        let mut sum_w2 = 0.0_f32;
        for w in self.weights.iter_mut() {
            *w *= inv;
            sum_w += *w;
            sum_w2 += *w * *w;
        }

        // 4. ESS.
        let ess = (sum_w * sum_w) / sum_w2;

        // 5. Conditional resample.
        let threshold = self.ess_threshold * (n as f32);
        let resampled = ess < threshold;

        if resampled {
            // ltsis fills `indices` with sampled indices into the
            // weight distribution. `weights` is read-only to it.
            ltsis::sample_indices_buffered(rng, self.weights, self.indices);
            // Gather: particles_curr[i] = particles_next[indices[i]].
            for i in 0..n {
                let j = self.indices[i] as usize;
                self.particles_curr[i] = self.particles_next[j].clone();
            }
            for w in self.weights.iter_mut() {
                *w = 1.0;
            }
        } else {
            // SIS step: the new "curr" is the just-propagated set;
            // weights carry over.
            core::mem::swap(&mut self.particles_curr, &mut self.particles_next);
        }

        Ok(StepResult {
            max_weight,
            sum_w,
            sum_w_squared: sum_w2,
            ess,
            resampled,
        })
    }
}
