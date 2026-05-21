//! # `ltbpf` — Linear-time Bayesian Particle Filter
//!
//! A **particle filter** is a recursive Bayesian estimator: it
//! maintains a cloud of weighted sample hypotheses ("particles") whose
//! empirical distribution approximates the posterior over a hidden
//! state that evolves in time. Given a stream of noisy observations,
//! the filter updates the cloud step by step.
//!
//! `ltbpf` implements the **bootstrap particle filter** (Gordon,
//! Salmond & Smith 1993), the standard textbook variant. It aims to
//! be small, fast, and easy to embed:
//!
//! - **Allocation-free, `no_std`-compatible.** Callers own every
//!   buffer the filter uses; the library never allocates. Runs on
//!   bare-metal targets like Cortex-M4F.
//! - **Model-agnostic.** Dynamics and observation likelihoods are
//!   plain closures, so the model can be nonlinear, discontinuous,
//!   or have non-Gaussian noise.
//! - **Linear-time resampling.** The expensive multinomial-resampling
//!   step uses [`ltsis`], which runs in O(n) where most implementations
//!   are O(n log n).
//!
//! ## When to reach for this
//!
//! A particle filter is the right tool when a Kalman filter is
//! awkward:
//!
//! - Your dynamics or sensors are **nonlinear** or **discontinuous**
//!   (proximity flags, range clipping, wrap-around bearings, …).
//! - Your noise is **non-Gaussian** (heavy-tailed, skewed,
//!   bounded, …).
//! - You don't have well-calibrated process- and measurement-noise
//!   covariances, and you don't want to tune them.
//!
//! If your problem is linear with Gaussian noise, a Kalman filter
//! will give the same answer faster. Use that instead.
//!
//! ## The two functions you write
//!
//! The entire model interface is two closures:
//!
//! - **`propagate(rng, &state) -> state`** samples one step of the
//!   transition kernel `p(x_t | x_{t-1})`. It should include process
//!   noise.
//!
//! - **`weight_update(&state, &obs) -> f32`** returns the (unnormalized)
//!   observation likelihood `p(obs | state)`, a non-negative finite
//!   number. Constant factors don't matter; the filter rescales
//!   between steps.
//!
//! Pass these to [`ParticleFilter::new`] along with four caller-owned
//! slices ([`Buffers`]), then call [`ParticleFilter::step`] once per
//! observation.
//!
//! ## Quick start
//!
//! A toy 1-D tracker: a hidden scalar `x` evolves as a random walk
//! and we observe `x + noise`.
//!
//! ```no_run
//! use ltbpf::{weighted_mean, Buffers, Coord, ParticleFilter};
//! use rand::rngs::SmallRng;
//! use rand::SeedableRng;
//! use rand_distr::{Distribution, Normal};
//!
//! const N: usize = 1000;
//! let mut rng = SmallRng::seed_from_u64(42);
//! let proc_noise = Normal::new(0.0_f32, 0.3).unwrap();
//! let obs_sigma  = 1.0_f32;
//!
//! // Four caller-owned buffers, all length N.
//! let mut p_curr: Vec<f32> =
//!     (0..N).map(|_| Normal::new(0.0, 2.0).unwrap().sample(&mut rng)).collect();
//! let mut p_next  = vec![0.0_f32; N];
//! let mut weights = vec![1.0_f32; N];
//! let mut indices = vec![0_u32;  N];
//!
//! let mut filter = ParticleFilter::new(
//!     Buffers {
//!         particles_curr: &mut p_curr,
//!         particles_next: &mut p_next,
//!         weights:        &mut weights,
//!         indices:        &mut indices,
//!     },
//!     // Dynamics: random walk.
//!     |rng, x: &f32| x + proc_noise.sample(rng),
//!     // Likelihood: Gaussian centered on the particle.
//!     |x: &f32, &obs: &f32| {
//!         let r = (obs - x) / obs_sigma;
//!         (-0.5 * r * r).exp()
//!     },
//! );
//!
//! for &obs in &[1.0_f32, 1.5, 2.1, 2.8, 3.6] {
//!     let result = filter.step(&mut rng, &obs).unwrap();
//!     let mean = weighted_mean(
//!         filter.particles(),
//!         filter.weights(),
//!         |x| [Coord::Linear(*x)],
//!     );
//!     let Coord::Linear(est) = mean[0] else { unreachable!() };
//!     println!("obs={obs}  est={est:.3}  ess={:.1}", result.ess);
//! }
//! ```
//!
//! For a richer example (2-D vehicle, GPS + IMU), see
//! `examples/vehicle.rs`.
//!
//! ## API at a glance
//!
//! - [`ParticleFilter`] — the filter struct.
//! - [`Buffers`] — the four caller-owned slices it borrows.
//! - [`ParticleFilter::step`] — advance by one observation; returns
//!   [`StepResult`] (diagnostics) or [`StepError`].
//! - [`weighted_mean`], [`map_particle`] — point estimators over the
//!   weighted particle cloud.
//! - [`Coord`] — per-dimension tag (`Linear` or `Angular`) for
//!   [`weighted_mean`].
//! - [`ResamplerKind`] — which `ltsis` backend to use for resampling.
//!
//! ## Effective sample size and adaptive resampling
//!
//! Between observations, the particle weights diverge: a few
//! particles end up carrying most of the mass while the rest contribute
//! little. The standard health metric is the **effective sample
//! size** (Kong, Liu & Wong 1994),
//!
//! ```text
//!     ESS = (Σ wᵢ)² / Σ wᵢ²
//! ```
//!
//! which equals `n` when all weights are equal and `1` when one
//! weight has all the mass. When `ESS` drops below `0.5·n` (the
//! default; see [`ParticleFilter::with_ess_threshold`]), the filter
//! resamples: it draws a new cloud of `n` particles in proportion to
//! their current weights, discarding the low-weight ones, and resets
//! all weights to `1`. This is the "Sequential Importance Resampling"
//! part of the algorithm.
//!
//! ## Cargo features
//!
//! Exactly one of:
//!
//! - **`std`** (default) — uses inherent `f32` math routines.
//! - **`libm`** — routes transcendentals through the `libm` crate.
//!   For bare-metal `no_std` targets.
//!
//! ## References
//!
//! - Gordon, Salmond & Smith (1993). The bootstrap particle filter.
//! - Kong, Liu & Wong (1994). Original ESS formula.
//! - Liu & Chen (1995). Adaptive-resampling threshold.
//! - Doucet & Johansen (2008). Tutorial reference.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(any(feature = "std", feature = "libm")))]
compile_error!(
    "Enable exactly one of the `std` or `libm` features so ltsis can \
     find its transcendental math."
);

use core::marker::PhantomData;

use rand::Rng;

/// Caller-owned scratch buffers borrowed by a [`ParticleFilter`].
///
/// All four slices must have the same nonzero length `n` (the
/// particle count). The filter borrows them for its lifetime and
/// never allocates.
///
/// Initialize `particles_curr` with samples from your prior before
/// constructing the filter; [`ParticleFilter::new`] takes care of
/// the rest (initial weights are set to `1.0`, the other two slices
/// are pure scratch).
///
/// After a successful [`ParticleFilter::step`], [`ParticleFilter::particles`]
/// and [`ParticleFilter::weights`] reflect the new posterior
/// approximation.
pub struct Buffers<'a, S> {
    /// The current particle cloud. Must be initialized from the
    /// prior before the first step.
    pub particles_curr: &'a mut [S],
    /// Scratch for propagated particles; the filter swaps this with
    /// `particles_curr` as needed. Contents on entry are ignored.
    pub particles_next: &'a mut [S],
    /// Particle weights. Reset to `1.0` by [`ParticleFilter::new`].
    pub weights: &'a mut [f32],
    /// Scratch used by the resampling step. Contents on entry are
    /// ignored; contents after a step are not part of the public API.
    pub indices: &'a mut [u32],
}

/// Diagnostic information returned by a successful
/// [`ParticleFilter::step`].
///
/// Callers that don't need diagnostics can discard the value with
/// `let _ = filter.step(...)?;`. Callers that do can read off the
/// scalars below without a second pass over the weights.
#[derive(Debug, Clone, Copy)]
pub struct StepResult {
    /// The largest unnormalized weight any particle had during this
    /// step (i.e. the largest value the user's `weight_update`
    /// produced, scaled by any weight carried in from earlier steps).
    /// Very small values warn that no particle is consistent with the
    /// latest observation — a useful "is the filter still tracking?"
    /// signal.
    pub max_weight: f32,

    /// Sum of the normalized particle weights. Always in `[1, n]`
    /// after a successful step.
    pub sum_w: f32,

    /// Sum of squared normalized particle weights. Together with
    /// `sum_w`, defines [`StepResult::ess`].
    pub sum_w_squared: f32,

    /// Effective sample size, `(Σ wᵢ)² / Σ wᵢ²`. Ranges from `1` (one
    /// particle carries all the mass) to `n` (weights are equal). The
    /// filter resamples when this drops below
    /// `ess_threshold · n` — see [`ParticleFilter::with_ess_threshold`].
    pub ess: f32,

    /// `true` if this step triggered a resample.
    pub resampled: bool,
}

/// Reasons [`ParticleFilter::step`] can fail at runtime.
///
/// Both variants mean "the filter no longer has a usable posterior
/// approximation" — typically caller-recoverable by reinitializing
/// the particle cloud from a prior.
///
/// On error the filter's internal buffers may have been partially
/// updated; treat the cloud as invalid until you've reinitialized.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum StepError {
    /// Every particle's weight has gone to zero — no hypothesis in
    /// the cloud is consistent with the latest observation. This
    /// is typical of sensor failure, model misspecification, or a
    /// "kidnapped robot" event. Recovery: reinitialize the cloud
    /// from a prior, widen your process noise, or signal upstream.
    AllWeightsZero,

    /// The user's `weight_update` closure returned a NaN, infinity,
    /// or negative value for at least one particle. Almost always a
    /// bug in the likelihood function.
    NonFiniteWeight,
}

/// A Sequential Importance Resampling (SIR) particle filter.
///
/// Built from [`Buffers`] (the four caller-owned slices) plus two
/// closures that define the user's model:
///
/// - `propagate: FnMut(&mut R, &S) -> S` — sample one step of the
///   state transition `p(x_t | x_{t-1})`, including process noise.
///
/// - `weight_update: FnMut(&S, &Obs) -> f32` — return the
///   unnormalized observation likelihood `p(obs | state)`, a
///   non-negative finite number.
///
/// The type parameters:
///
/// - `S` — particle state. Must be [`Clone`]. Keep it small;
///   resampling copies particles around. For heavy state, store
///   `Arc<HeavyThing>` as `S` so cloning is a refcount bump.
/// - `R` — RNG type, e.g. `rand::rngs::SmallRng`.
/// - `Obs` — observation type, passed to `weight_update`.
/// - `Prop`, `Weigh` — closure types, inferred at construction.
///
/// See the crate-level docs for a complete example.
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
    resampler: ResamplerKind,
    _phantom: PhantomData<fn(&mut R, &Obs)>,
}

/// Which `ltsis` entry point [`ParticleFilter`] uses for resampling.
///
/// Pick this with [`ParticleFilter::with_resampler`]. Both choices
/// produce a statistically equivalent posterior; they differ only in
/// speed and which scratch they touch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ResamplerKind {
    /// Default. Uses `ltsis::sample_indices_buffered` — the bit-stashed
    /// linear-time resampler. About 30% faster than `Streaming` on x86;
    /// more on Cortex-M4F.
    #[default]
    Buffered,
    /// Uses `ltsis::sample_indices`, the iterator form. Slightly slower
    /// than `Buffered`; useful for the rare memory-constrained
    /// `no_std` setting where every byte matters (the `indices`
    /// buffer in [`Buffers`] is still required for API uniformity
    /// but is not touched in this mode).
    Streaming,
}

impl<'a, S, R, Obs, Prop, Weigh> ParticleFilter<'a, S, R, Obs, Prop, Weigh>
where
    R: Rng + ?Sized,
    Prop: FnMut(&mut R, &S) -> S,
    Weigh: FnMut(&S, &Obs) -> f32,
    S: Clone,
{
    /// Construct a filter from caller-owned buffers and the two
    /// model closures.
    ///
    /// Initialize `buffers.particles_curr` from your prior before
    /// calling this; the other three slices are scratch (their
    /// contents are overwritten or initialized as needed).
    /// `weights` is reset to all-ones here.
    ///
    /// # Panics
    ///
    /// Panics if any buffer length differs from
    /// `particles_curr.len()`, or if `particles_curr` is empty.
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
            resampler: ResamplerKind::Buffered,
            _phantom: PhantomData,
        }
    }

    /// Select the resampling backend. See [`ResamplerKind`].
    pub fn with_resampler(mut self, kind: ResamplerKind) -> Self {
        self.resampler = kind;
        self
    }

    /// Set the effective-sample-size threshold for adaptive
    /// resampling. The filter resamples when `ess < threshold · n`.
    /// Default is `0.5`.
    ///
    /// - `1.0` forces a resample on every step (Gordon-Salmond-Smith
    ///   bootstrap filter, the original 1993 formulation).
    /// - `0.0` disables resampling entirely (pure importance
    ///   sampling — useful only for diagnostics).
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
    /// Returns diagnostics for the step (see [`StepResult`]). On a
    /// filter-killing pathology — every particle's weight zero, or
    /// the user's `weight_update` producing NaN/inf/negative — returns
    /// [`StepError`] instead and leaves the cloud in an unspecified
    /// state.
    pub fn step(&mut self, rng: &mut R, obs: &Obs) -> Result<StepResult, StepError> {
        // Standard SIR step:
        //   1. Propagate each particle via the user's transition kernel.
        //   2. Multiply weights by the user's observation likelihood.
        //   3. Normalize weights so the maximum is 1.0. (Working in
        //      linear space rather than log space saves a per-particle
        //      expf; the renormalize step keeps weights in [0, 1] and
        //      well clear of f32 underflow as long as adaptive
        //      resampling fires regularly enough.)
        //   4. Compute ESS.
        //   5. If ess < threshold * n, resample (via ltsis) and reset
        //      weights to 1.0. Otherwise swap curr/next so the weights
        //      carry forward — between resamples, the filter is just
        //      importance sampling.
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
            match self.resampler {
                ResamplerKind::Buffered => {
                    // ltsis fills `indices` with sampled indices into
                    // the weight distribution; `weights` is read-only.
                    ltsis::sample_indices_buffered(rng, self.weights, self.indices);
                    for i in 0..n {
                        let j = self.indices[i] as usize;
                        self.particles_curr[i] = self.particles_next[j].clone();
                    }
                }
                ResamplerKind::Streaming => {
                    // Drive the gather directly from the iterator —
                    // indices are yielded in ascending order, one per
                    // gather slot. The `indices` buffer is left
                    // untouched.
                    let it = ltsis::sample_indices(rng, self.weights, n as u32);
                    for (i, j) in it.enumerate() {
                        self.particles_curr[i] = self.particles_next[j as usize].clone();
                    }
                }
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

// ===========================================================================
// Estimators
// ===========================================================================

/// Per-dimension tag for state coordinates projected by
/// [`weighted_mean`].
///
/// Use `Linear` for ordinary real-valued dimensions (position,
/// velocity, …) and `Angular` for angles in radians, which are
/// averaged across the `±π` wrap correctly. The output of
/// [`weighted_mean`] is an array of `Coord` values with the same
/// per-dimension variants you passed in.
///
/// Marked `#[non_exhaustive]` so additional coordinate kinds may be
/// added in the future without breaking the public API.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum Coord {
    /// Real-valued dimension. Averaged as an ordinary arithmetic mean.
    Linear(f32),
    /// Angle in radians, conventionally in `(-π, π]`. Averaged so
    /// that, e.g., the mean of `+3.0` and `-3.0` is near `±π` rather
    /// than near `0`.
    Angular(f32),
}

/// Weighted centroid of the particle cloud.
///
/// Each particle is projected through `project` to a `[Coord; D]`
/// array, and each output dimension is the weighted mean of the
/// corresponding input dimension. `Coord::Linear` dimensions use the
/// ordinary arithmetic weighted mean. `Coord::Angular` dimensions
/// average across the `±π` wrap correctly.
///
/// # Example
///
/// ```
/// # use ltbpf::{weighted_mean, Coord};
/// #[derive(Clone)]
/// struct Robot { x: f32, y: f32, heading: f32 }
///
/// let particles = vec![
///     Robot { x: 1.0, y: 2.0, heading:  3.10 },
///     Robot { x: 1.2, y: 2.1, heading: -3.10 },  // wraps near ±π
/// ];
/// let weights = vec![1.0_f32, 1.0];
/// let mean = weighted_mean(&particles, &weights, |r| {
///     [Coord::Linear(r.x), Coord::Linear(r.y), Coord::Angular(r.heading)]
/// });
/// // mean[0] = Linear(1.1), mean[1] = Linear(2.05),
/// // mean[2] = Angular(~±π) — not near 0!
/// ```
///
/// # Multimodal posteriors
///
/// Any single-point estimator — `weighted_mean` included — is
/// **misleading** when the posterior has multiple well-separated
/// modes. The weighted centroid of two clusters is the (unphysical)
/// midpoint between them. If you suspect multimodality, use
/// [`map_particle`] for a quick check or reach for a clustering tool
/// — `ltbpf` deliberately does not ship one.
///
/// # Caller contract
///
/// `project` should return the same sequence of `Coord` variants for
/// every particle. The library does not check this; mixing variants
/// across particles will silently produce nonsense.
///
/// # Panics
///
/// Panics if `particles` and `weights` have different lengths, if
/// `particles` is empty, or if the total weight is non-positive.
pub fn weighted_mean<S, const D: usize>(
    particles: &[S],
    weights: &[f32],
    project: impl Fn(&S) -> [Coord; D],
) -> [Coord; D] {
    // The angular case uses an online weighted mean of shortest-rotation
    // displacements: each new particle's angle is reduced to (-π, π]
    // relative to the current running mean and added in with weight
    // wᵢ / Σ_{j≤i} wⱼ. No sin/cos/atan2 on the hot path; exactly
    // correct and order-independent for any unimodal angular cloud
    // (one where the running mean stays within π of every sample).
    assert_eq!(
        particles.len(),
        weights.len(),
        "particle/weight length mismatch"
    );
    assert!(
        !particles.is_empty(),
        "weighted_mean requires at least one particle"
    );

    const PI: f32 = core::f32::consts::PI;
    const TWO_PI: f32 = 2.0 * PI;

    let first_coords = project(&particles[0]);
    let mut sum_linear = [0.0_f32; D];
    let mut running_ang = [0.0_f32; D];

    let w0 = weights[0];
    for k in 0..D {
        match first_coords[k] {
            Coord::Linear(x) => sum_linear[k] = w0 * x,
            Coord::Angular(theta) => running_ang[k] = theta,
        }
    }
    let mut sum_w = w0;

    for i in 1..particles.len() {
        let w = weights[i];
        let coords = project(&particles[i]);
        let sum_w_new = sum_w + w;
        let scale = if sum_w_new > 0.0 { w / sum_w_new } else { 0.0 };
        for k in 0..D {
            match coords[k] {
                Coord::Linear(x) => sum_linear[k] += w * x,
                Coord::Angular(theta) => {
                    let mut d = theta - running_ang[k];
                    while d > PI {
                        d -= TWO_PI;
                    }
                    while d <= -PI {
                        d += TWO_PI;
                    }
                    running_ang[k] += scale * d;
                }
            }
        }
        sum_w = sum_w_new;
    }

    assert!(
        sum_w > 0.0,
        "weighted_mean requires positive total weight, got {sum_w}"
    );

    let mut out = first_coords;
    for k in 0..D {
        match first_coords[k] {
            Coord::Linear(_) => out[k] = Coord::Linear(sum_linear[k] / sum_w),
            Coord::Angular(_) => {
                let mut m = running_ang[k];
                while m > PI {
                    m -= TWO_PI;
                }
                while m <= -PI {
                    m += TWO_PI;
                }
                out[k] = Coord::Angular(m);
            }
        }
    }
    out
}

/// Return a clone of the particle with the largest weight — a rough
/// "MAP" estimate over the discrete particle approximation.
///
/// Useful as a sanity check or for posteriors where [`weighted_mean`]
/// would be misleading (multiple modes). Discards the within-mode
/// detail that `weighted_mean` would capture.
///
/// Ties are broken by lower index.
///
/// # Panics
///
/// Panics if `particles` and `weights` have different lengths or if
/// `particles` is empty.
pub fn map_particle<S: Clone>(particles: &[S], weights: &[f32]) -> S {
    assert_eq!(
        particles.len(),
        weights.len(),
        "particle/weight length mismatch"
    );
    assert!(
        !particles.is_empty(),
        "map_particle requires at least one particle"
    );
    let mut best = 0usize;
    let mut best_w = weights[0];
    for (i, &w) in weights.iter().enumerate().skip(1) {
        if w > best_w {
            best = i;
            best_w = w;
        }
    }
    particles[best].clone()
}
