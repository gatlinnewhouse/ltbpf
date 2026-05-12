# ltbpf: linear-time Bayesian Particle Filtering
Bart Massey and Claude Code 2026

[![Crates.io](https://img.shields.io/crates/v/ltbpf.svg)](https://crates.io/crates/ltbpf)
[![Documentation](https://docs.rs/ltbpf/badge.svg)](https://docs.rs/ltbpf)
[![CI](https://github.com/BartMassey/ltbpf/actions/workflows/ci.yml/badge.svg)](https://github.com/BartMassey/ltbpf/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/crates/l/ltbpf.svg)](#license)

A small Rust crate for **Bayesian particle filtering**. Given a hidden
state that evolves over time and a stream of noisy observations of
it, a particle filter gives you a running estimate of where the state
probably is.

`ltbpf` is built for the cases where a Kalman filter is awkward —
nonlinear dynamics, discontinuous sensors, non-Gaussian noise,
or "I don't want to tune covariance matrices." It is small, fast,
allocation-free, and works on embedded targets (Cortex-M4F-class
microcontrollers and up).

## Why a particle filter?

A particle filter represents a probability distribution by a cloud of
sampled hypotheses ("particles"). Each step it does two things:

1. **Move each particle forward in time** using your model of how
   the state evolves (`propagate`).
2. **Re-weight the particles** by how plausible each one looks given
   the latest observation (`weight_update`), occasionally throwing
   out low-weight ones and duplicating high-weight ones
   (*resampling*).

The weighted cloud at any point is your posterior over the state.

Where a Kalman filter wants `Q` and `R` covariance matrices and
assumes linearity plus Gaussian noise, a particle filter wants two
functions:

| If you have…                         | …a Kalman filter wants | …`ltbpf` wants |
|--------------------------------------|------------------------|----------------|
| A way to step the state forward      | linear matrix `F` + covariance `Q` | a `propagate` closure |
| A way to score an observation        | linear matrix `H` + covariance `R` | a `weight_update` closure |
| Nonlinear dynamics                   | needs the EKF/UKF dance | works as-is |
| Non-Gaussian or weird sensor noise   | gives wrong answers     | works as-is |
| Lots of compute budget               | not strictly necessary  | a particle cloud is more compute |

If your problem is linear with Gaussian noise, prefer a Kalman filter:
it'll give the same answer faster. If it isn't, read on.

## Installing

```toml
[dependencies]
ltbpf = "0.1"
rand = "0.10"
rand_distr = "0.6"
```

The library is `no_std`-compatible. For bare-metal targets, disable
default features and enable `libm`:

```toml
[dependencies.ltbpf]
version = "0.1"
default-features = false
features = ["libm"]
```

## A complete example

Tracking a hidden scalar `x` that drifts randomly, given noisy
measurements of it:

```rust
use ltbpf::{weighted_mean, Buffers, Coord, ParticleFilter};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

const N: usize = 1000; // number of particles

let mut rng = SmallRng::seed_from_u64(42);
let proc_noise = Normal::new(0.0_f32, 0.3).unwrap();
let obs_sigma  = 1.0_f32;

// 1. Four caller-owned buffers, all the same length.
let mut p_curr: Vec<f32> =
    (0..N).map(|_| Normal::new(0.0, 2.0).unwrap().sample(&mut rng))
          .collect();
let mut p_next  = vec![0.0_f32; N];
let mut weights = vec![1.0_f32; N];
let mut indices = vec![0_u32;  N];

// 2. Build the filter: pass in the buffers and your two model
//    closures.
let mut filter = ParticleFilter::new(
    Buffers {
        particles_curr: &mut p_curr,
        particles_next: &mut p_next,
        weights:        &mut weights,
        indices:        &mut indices,
    },
    // propagate: how does a particle move from one step to the next?
    |rng, x: &f32| x + proc_noise.sample(rng),

    // weight_update: how plausible is this particle given the latest
    // observation? (Up to a normalization constant — `ltbpf`
    // rescales between steps, so absolute scale is irrelevant.)
    |x: &f32, &obs: &f32| {
        let r = (obs - x) / obs_sigma;
        (-0.5 * r * r).exp() // Gaussian likelihood
    },
);

// 3. Feed in observations one at a time.
for &obs in &[1.0_f32, 1.5, 2.1, 2.8, 3.6] {
    let result = filter.step(&mut rng, &obs).unwrap();

    // 4. Get a point estimate from the cloud.
    let centroid = weighted_mean(
        filter.particles(), filter.weights(),
        |x| [Coord::Linear(*x)],
    );
    let Coord::Linear(estimate) = centroid[0] else { unreachable!() };

    println!("obs={obs}  estimate={estimate:.3}  ess={:.1}", result.ess);
}
```

Output (approximate):

```
obs=1.0  estimate=0.51  ess=786.4
obs=1.5  estimate=0.98  ess=920.1
obs=2.1  estimate=1.47  ess=918.6
obs=2.8  estimate=2.11  ess=912.8
obs=3.6  estimate=2.83  ess=901.0
```

The estimate trails the observations because the filter is fusing
each measurement with the running posterior; this is the point of a
filter.

## The two functions you write

**`propagate(rng, &state) -> state`** samples one timestep of how the
state changes. Include process noise here — the cloud needs to spread
out a little each step so it can react to new evidence.

**`weight_update(&state, &obs) -> f32`** returns the likelihood of
the observation given that state, up to a constant. The standard
choice is a Gaussian centered on the particle's predicted observation:

```rust
fn likelihood(particle: &State, obs: &Obs) -> f32 {
    let r = (obs.value - predicted(particle)) / sigma;
    (-0.5 * r * r).exp()
}
```

For more complicated sensors, multiply likelihoods or build whatever
distribution fits your model. The number just has to be non-negative
and finite.

## Getting an estimate

The filter holds the weighted cloud; you turn that cloud into a
single number (or vector) with one of two estimators:

- **`weighted_mean(particles, weights, project)`** — the weighted
  centroid. Each dimension is tagged `Coord::Linear` or
  `Coord::Angular`; angular dimensions are averaged correctly across
  the `±π` wrap.

- **`map_particle(particles, weights)`** — returns the highest-weight
  particle. Less smooth than `weighted_mean`, but it returns *one of
  the modes* rather than averaging across them — handy as a sanity
  check when you suspect multimodality.

Don't use `weighted_mean` on a sharply multimodal posterior — the
"centroid of two clusters" is the midpoint between them, which is
usually nowhere any particle actually is. (`ltbpf` doesn't include a
clustering tool. If you need one, see the discussion in the design
plan.)

## Tuning knobs

```rust
let filter = ParticleFilter::new(buffers, propagate, likelihood)
    .with_ess_threshold(0.5)          // default
    .with_resampler(ResamplerKind::Buffered);  // default
```

- **`with_ess_threshold(t)`** — when the effective sample size drops
  below `t · n`, the filter resamples. The default `0.5` is the
  Liu & Chen (1995) recommendation and works for most problems.
- **`with_resampler(kind)`** — `Buffered` (the default, faster) or
  `Streaming` (saves a bit of scratch on `no_std` targets).

## What it gives you back

`filter.step(&mut rng, &obs)` returns either:

- **`Ok(StepResult)`** — diagnostics for the step: the effective
  sample size, whether the filter resampled, the maximum weight
  (useful for detecting "is the filter still tracking?"), …
- **`Err(StepError)`** — either every particle's weight went to zero
  (the cloud has lost the state, e.g. a sensor failure) or your
  `weight_update` produced NaN/inf/negative (a bug in the
  likelihood). In either case the cloud is no longer trustworthy —
  reinitialize from the prior or signal upstream.

## Worked examples in the repo

```sh
cargo run --release --example vehicle > out.csv
cargo run --release --example compare_resamplers > resamplers.csv
```

- **`vehicle`** — a 2-D vehicle with noisy GPS (position) and IMU
  (velocity) sensors. Writes a CSV of truth, estimate, ESS, and
  tracking error per timestep. ~90 ms for 1 000 particles × 1 000
  steps on a desktop x86.

  To visualize the result, either:

  ```sh
  python3 examples/plot.py out.csv               # matplotlib
  gnuplot -c examples/plot.gp out.csv            # gnuplot
  ```

  Both produce a three-panel figure: 2-D tracks (truth vs estimate),
  tracking error over time, and effective sample size over time.
  Pass an extra path argument to either script to save the figure
  to a PNG instead of opening a window.

  ![Vehicle demo output](vehicle.png)

  *A representative run: 1 000 particles, 1 000 steps, seed
  `0xC0FFEE`. Top panel — the truth trajectory (blue) and the
  filter's weighted-mean estimate (orange), nearly overlapping after
  the initial convergence. Middle — Euclidean tracking error, which
  drops from ~4 m at startup to ~0.3 m once the cloud has locked
  on. Bottom — effective sample size; the filter resamples when
  this dips, which keeps it bouncing in a healthy range below
  N = 1 000.*

- **`compare_resamplers`** — the same vehicle filter run with three
  different resamplers (`ltsis` buffered, `ltsis` streaming, and a
  textbook prefix-sum + binary-search baseline) at several values of
  N. Demonstrates the `ltsis` speedup over the standard
  implementation — about 2.3× at N = 30 000.

## Benchmarks

```sh
cargo bench
```

`benches/bench.rs` uses [Divan] for per-iteration timing with outlier
rejection. It measures one full `ParticleFilter::step` call on the
2-D vehicle model from `examples/vehicle.rs`, separately for each
[`ResamplerKind`], and the two estimators (`weighted_mean`,
`map_particle`).

Representative numbers from one host (AMD Ryzen 9 3900X, x86-64, single
thread, release build, `SmallRng`/Xoshiro256++, fastest run reported):

**`ParticleFilter::step` (one full step: propagate + weight + ESS +
resample-or-swap)**

| N        | Buffered  | Streaming | ns / particle |
|----------|-----------|-----------|---------------|
|    300   |   5.4 µs  |   5.3 µs  |     ~18       |
|  1 000   |  18.1 µs  |  18.1 µs  |     ~18       |
|  3 000   |  54.7 µs  |  54.1 µs  |     ~18       |
| 10 000   |  181 µs   |  182 µs   |     ~18       |

Per-step cost is essentially linear in N, as expected (every phase of
the filter is O(n)). Buffered and Streaming come out within noise of
each other on x86 — at these sizes the resampler is a small fraction
of step time, which is dominated by the model's `propagate` closure
(two `Normal` draws per particle per step). The difference shows up
in the standalone resampler benchmark below.

**Estimators (single pass over an `n`-particle weighted cloud)**

| N        | `weighted_mean` (4-D linear) | `map_particle` |
|----------|------------------------------|----------------|
|    300   |   216 ns                     |   148 ns       |
|  1 000   |   711 ns                     |   444 ns       |
|  3 000   |   2.1 µs                     |   1.3 µs       |
| 10 000   |   7.1 µs                     |   4.4 µs       |

Roughly 0.7 ns per particle per coordinate for `weighted_mean`; 0.4 ns
per particle for `map_particle`'s simple max-scan.

**Resampler comparison** (separate example, not the divan harness):

```sh
cargo run --release --example compare_resamplers > resamplers.csv
```

Compares the `ltsis` buffered and streaming resamplers against a
textbook prefix-sum + binary-search baseline, inside the same vehicle
filter. On the same host, at N = 30 000: ltsis buffered ~1.27 ms/step,
ltsis streaming ~1.51 ms/step, naive ~2.98 ms/step — a ~2.3× speedup
for the buffered variant.

[Divan]: https://docs.rs/divan

## Running the tests

```sh
cargo test --release
```

The test suite has three layers:

- **`tests/mechanics.rs`** — filter bookkeeping (weight normalization,
  ESS, resample reset, the streaming resampler, error variants).
- **`tests/estimators.rs`** — `weighted_mean` (linear 2-D, angular
  near 0, angular straddling ±π, asymmetric, permutation invariance)
  and `map_particle`.
- **`tests/convergence_kalman.rs`** — runs the BPF and an analytic
  Kalman filter on the same 1-D linear-Gaussian random walk across
  50 trials; the BPF mean tracks Kalman within ~0.06 Kalman SDs.

## References

- Gordon, Salmond & Smith (1993). The bootstrap particle filter (the
  algorithm this crate implements).
- Kong, Liu & Wong (1994). The effective sample size formula.
- Liu & Chen (1995). Adaptive-resampling threshold.
- Doucet & Johansen (2008). *A tutorial on particle filtering and
  smoothing: fifteen years later.* The standard tutorial reference.

For the linear-time resampling primitive: [`ltsis`].

[`ltsis`]: https://github.com/BartMassey/ltsis

## License

Dual-licensed under either of:

- [MIT License](LICENSE-MIT)
- [Apache License, Version 2.0](LICENSE-APACHE)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution
intentionally submitted for inclusion in the work by you, as
defined in the Apache-2.0 license, shall be dual licensed as
above, without any additional terms or conditions.
