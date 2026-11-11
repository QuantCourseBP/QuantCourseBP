# Finite-difference Greeks in `Pricer` (Assignment 1)

## What was changed
* `src/enums.py`: two new Enums - `FiniteDifferenceMethod` (FORWARD, BACKWARD, CENTRAL) and `BumpType` (ABSOLUTE, RELATIVE).
* `src/pricer.py`: two new class variables on `Pricer`: `fd_method` and `bump_type` (plus `spot_bump_size`, the size of the bump:
  a fraction of spot if RELATIVE, currency units if ABSOLUTE). `calc_delta` / `calc_gamma` (BUMP) implement all 3 x 2 combinations.
  * delta: forward `(V(S+h)-V(S))/h`, backward `(V(S)-V(S-h))/h`, central `(V(S+h)-V(S-h))/(2h)`
  * gamma: forward `(V(S+2h)-2V(S+h)+V(S))/h^2`, backward `(V(S)-2V(S-h)+V(S-2h))/h^2`, central `(V(S+h)-2V(S)+V(S-h))/h^2`
* `test/test_fd_greeks.py`: checks all 6 combinations against the analytic Greeks (calls and puts). All 220 repo tests pass.
* `homeworks/29481ec_greeks/greeks_study.py`: study on 108 European options (3 stocks x call/put x strike/spot 0.8,1,1.2 x expiry 0.25,1,2),
  60 bump sizes per method and bump type.

## Results (median error over all options; machine epsilon ~ 2.2e-16)
| Scheme | Greek | Optimal relative bump h/S | Error at optimum | Theory |
|---|---|---|---|---|
| Forward | delta | 1e-8 | 6e-9 (abs) | sqrt(eps) = 1.5e-8 |
| Backward | delta | 7e-9 | 6e-9 (abs) | sqrt(eps) |
| Central | delta | 4e-6 | 1e-11 (abs) | eps^(1/3) = 6e-6 |
| Forward / Backward | gamma | 4e-6 | 1e-5 (relative) | eps^(1/3) |
| Central | gamma | 1.1e-4 | 1.2e-8 (relative) | eps^(1/4) = 1.2e-4 |

Truncation error (grows with h) and rounding error (grows as h shrinks) give the V-shaped curves in `fig1_error_vs_bump.png`;
the minimum sits where the two balance, matching the theory above. Central differences are second-order accurate, so they are
several orders of magnitude better than forward/backward, which are first-order for delta (and for gamma too, since the one-sided
second difference is only O(h)).

## Absolute vs relative bump
Both work if tuned, but the best *absolute* bump scales with the spot (central delta: about 5.6e-4 for spot 130-177 and 1.1e-3 for
spot 251), while the best *relative* bump is almost the same for all stocks (3e-6 to 6e-6, `fig2_absolute_vs_relative.png`).
A relative bump therefore needs one setting for the whole book; an absolute bump needs re-tuning for every underlying and
breaks for very low or high priced underlyings. Rounding error depends on the relative size h/S, so the relative convention is the natural one.

## Recommended bump sizes (relative to spot)
| Scheme | Delta | Gamma |
|---|---|---|
| Central (recommended) | 1e-5 | 1e-4 |
| Forward / Backward | 1e-8 | 4e-6 |

A single safe choice for both with central differences is `h/S = 1e-4` (delta error ~3e-9, gamma relative error ~1e-8).
For comparison, the library's former default of 1% gives a central delta error of 3e-5 and gamma relative error of 7e-5, and forward/backward
errors of 4e-3 (delta) and 2e-2 (relative gamma). For absolute bumps use `h = (relative size) x spot`, e.g. about 1e-2 currency units for S=100 (gamma, central).
Caveat: this is for smooth Black-Scholes prices; with Monte Carlo/PDE/tree prices the noise is much larger than rounding error,
so the bump must be much larger (or common random numbers used).
