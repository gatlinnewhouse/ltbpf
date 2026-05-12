# `ltbpf`

Linear-time Bayesian Particle Filter for Rust. A small,
allocation-free, `no_std`-clean implementation of the standard
Sequential Importance Resampling (SIR) bootstrap particle filter
(Gordon, Salmond & Smith 1993), with O(n) multinomial resampling
delegated to [`ltsis`](https://github.com/BartMassey/ltsis).

```rust
let mut filter = ParticleFilter::new(
    Buffers { particles_curr, particles_next, weights, indices },
    |rng, state| propagate_one_step(rng, state),
    |state, obs| likelihood(state, obs),
);

for obs in observations {
    let StepResult { ess, resampled, .. } =
        filter.step(&mut rng, &obs)?;
    let centroid = weighted_mean(
        filter.particles(), filter.weights(),
        |s| [Coord::Linear(s.x), Coord::Linear(s.y)],
    );
    // ... use the estimate ...
}
```

## Why a particle filter (vs a Kalman filter)

`ltbpf` is for the cases where Kalman is awkward:

- **Less tuning.** A Kalman filter wants process- and
  measurement-noise covariance matrices `Q` and `R`. A BPF wants a
  sample-from-the-transition function and a
  likelihood-of-observation function. Both are usually natural to
  write directly, with no covariance estimation step in between.
- **Nonlinear dynamics or observations.** Kalman needs linearity;
  the Extended Kalman Filter linearizes but loses accuracy when the
  nonlinearity is strong. BPF handles arbitrary nonlinearity for
  free — your `propagate` and `weight_update` closures can compute
  anything.
- **Discontinuous sensors.** Proximity flags, range clipping,
  wrap-around bearings, occlusion gates — all break Kalman's
  smoothness assumptions. BPF doesn't care; the likelihood function
  evaluates whatever sensor model you have.
- **Non-Gaussian noise.** Heavy-tailed, skewed, or
  compactly-supported noise distributions break Kalman's
  Gaussian-conjugate update. BPF takes whatever pdf you code up.

If your model is genuinely linear-Gaussian, use a Kalman filter:
it'll give the same answer with less compute. This crate's bundled
[vehicle demo](examples/vehicle.rs) is in fact linear-Gaussian, so
that the test suite can compare BPF output against an analytic
Kalman filter on the same observations (see
[`tests/convergence_kalman.rs`](tests/convergence_kalman.rs)).

## What's in the box

- **`ParticleFilter`** — the SIR loop. Holds references to
  caller-owned buffers; no allocation; works the same on a
  workstation and a Cortex-M4F. Adaptive resampling triggered by
  effective sample size (default threshold `ESS < 0.5n`, after Liu
  & Chen 1995); switchable resampling backend (`ResamplerKind::{Buffered,
  Streaming}`).
- **`weighted_mean`** — weighted centroid estimator with
  per-dimension `Coord::{Linear, Angular}` tagging. Angular
  dimensions use an online shortest-rotation mean (no `sin`/`cos`/
  `atan2` on the hot path).
- **`map_particle`** — argmax-of-weights, the discrete-MAP
  estimate.
- **`Buffers`** — the four caller-owned slices: two particle
  arrays (current and scratch), weights, and resample-index scratch.

Out of scope (deliberately): multimodal state estimation via
spatial clustering, particle rejuvenation / roughening,
Rao-Blackwellized filters, smoothing, joint parameter-state
inference. See [`ltbpf-plan.md`](ltbpf-plan.md) for the full
rationale.

## Features

```toml
[dependencies]
ltbpf = { git = "https://github.com/BartMassey/ltbpf" }
```

Cargo features (exactly one of `std`/`libm` must be enabled):

- `std` (default) — pulls `ltsis/std`, uses inherent
  `f32`/`f64` math.
- `libm` — routes transcendental math through the `libm`
  crate. Use this on bare-metal `no_std` targets.

The library proper is `no_std`-clean; the bundled examples are
gated behind `std` (they use `println!` and `Vec`).

## Examples

```
cargo run --release --example vehicle [n] > out.csv
cargo run --release --example compare_resamplers > resamplers.csv
```

- **`vehicle`** — 2D near-constant-velocity vehicle with noisy
  GPS + IMU. Mirrors Figure 4 of Massey (ICASSP 2008). 7-column
  CSV: `step, truth_x, truth_y, est_x, est_y, ess, err`.
- **`compare_resamplers`** — drives the vehicle BPF loop with
  three resamplers (`ltsis::sample_indices_buffered`,
  `ltsis::sample_indices`, and a textbook prefix-sum +
  binary-search baseline) at several values of N. CSV columns:
  `n, resampler, total_ms, per_step_us`. At N = 30,000 on a
  desktop x86: ltsis buffered ~1.27 ms/step, naive ~2.98 ms/step.

## Tests

```
cargo test --release
```

Three layers:

- `tests/mechanics.rs` — bookkeeping invariants: ESS tabular
  checks, weight normalization, resample reset, SIS carry-over,
  `StepError` paths, streaming resampler.
- `tests/estimators.rs` — `weighted_mean` (linear 2D centroid,
  angular cluster near 0, angular ±π straddle, asymmetric weights,
  permutation invariance) and `map_particle` (argmax + tiebreak).
- `tests/convergence_kalman.rs` — 50 fixed-seed trials of 100
  steps on a 1D linear-Gaussian random walk; verifies the BPF
  weighted mean tracks the analytic Kalman mean to within 0.5
  Kalman SDs averaged across the run (achieved value with N=500
  particles: ~0.06 Kalman SDs).

## Implementation notes

- **Weights are kept in linear space**, max-normalized every step
  so the largest weight is exactly `1.0`. This trades one
  normalization pass per step for the per-particle `expf` cost of
  working in log space — worthwhile on Cortex-M4F (~50–100 cycles
  per `expf`); the dynamic range concern is mitigated by adaptive
  resampling plus the running max-normalize.
- **`step()` returns `Result<StepResult, StepError>`.**
  `StepResult` carries `max_weight`, `sum_w`, `sum_w_squared`,
  `ess`, and `resampled`; `StepError` is `AllWeightsZero` or
  `NonFiniteWeight`. Buffer-length mismatch and zero-particle
  cases are panics (programmer errors).
- **`S: Clone`, not `Copy`.** Gather costs one clone per
  resampled particle (unavoidable — multiple output slots can come
  from the same input slot). For heavy state, use `Arc<HeavyThing>`
  so clone is a refcount bump.

## Citations

For the SIR loop and ESS:

- Gordon, Salmond & Smith (1993), "Novel approach to nonlinear/
  non-Gaussian Bayesian state estimation," *IEE Proceedings F*,
  140(2), 107–113.
- Kong, Liu & Wong (1994), "Sequential imputations and Bayesian
  missing data problems," *JASA* 89(425), 278–288. Original ESS
  formula.
- Liu & Chen (1995/1998), adaptive-resampling threshold.
- Del Moral, Doucet & Jasra (2012), "On adaptive resampling
  strategies for sequential Monte Carlo methods," *Bernoulli*
  18(1), 252–278.
- Doucet & Johansen (2008), "A tutorial on particle filtering and
  smoothing: fifteen years later," in *Handbook of Nonlinear
  Filtering*, Oxford UP. The standard tutorial reference.

For the linear-time resampling primitive:

- See [`ltsis`](https://github.com/BartMassey/ltsis).

## License

MIT OR Apache-2.0.
