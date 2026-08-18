# conserved-domain-hyperuniformity

Analysis code, data and manuscript for **"Conserved dynamics make the solute hyperuniform, not the domains"**.

In a phase-separating system with a locally conserved order parameter, the concentration field is hyperuniform ($S_c(k)\sim k^4$) while the point pattern of domain centroids is **not**. Weighting the same points by domain solute mass restores hyperuniformity. This repository contains everything needed to reproduce that result and the variance-budget analysis that explains it.

---

## Headline numbers

$1024^2$ periodic grid, $c_0 = 0.30$, 4 independent runs, 2000 coarsening steps, $2736 \pm 20$ domains per configuration.

| object / null | Clark–Evans $R$ | $S(k\to0)$ | $\alpha$ | $\sigma^2$ slope |
|---|---|---|---|---|
| domain centroids (counts) | 1.73 (z = +73) | 0.143 | −0.14 ± 0.04 | 1.80 |
| same points, mass-weighted | — | **0.021** | +0.40 ± 0.09 | — |
| concentration field | — | — | **+3.81 ± 0.08** | — |
| CSR control (fixed *N*) | 0.99 | 1.14 | −0.05 | **2.02** |
| RSA hard core | 1.42 | 0.24 | +0.24 | 1.79 |
| perturbed lattice | 1.45 | 0.017 | +2.02 | 0.93 |

Variance budget at the widest window ($R = 170.7$ cells $\approx$ 8 domain spacings): independent domains of the observed size distribution would give a solute-mass variance of **100.3** in units of one domain; the measured value is **12.2**. Permuting the masses over the same positions returns the budget to its baseline (ratio 0.99–1.09), locating the entire effect in mass–position coupling.

---

## Reproducing

Requires Python 3.10+, `numpy`, `scipy`, `matplotlib`. No other dependencies; no network access needed.

```bash
# 1. simulate.  Checkpointed: run the same command repeatedly until it
#    reports every run finished (~90 s per seed per 420-step chunk).
python src/hyperuniformity_analysis.py --n 1024 --seeds 4 --steps 2000 \
       --snaps 1000,2000 --cache f1024.npz --simulate-only --chunk 420

# 2. analyse: statistics, nulls, journal figures, JSON
python src/hyperuniformity_analysis.py --n 1024 --seeds 4 --steps 2000 \
       --snaps 1000,2000 --stages 2000 --no-scan --cache f1024.npz \
       --compare data/hyperuniformity_512.json

# 3. mass autocorrelation, mark-permutation attribution, per-domain export
python src/mass_autocorrelation.py --cache f1024.npz --stage 2000 --seeds 4

# redraw figures only, from the stored JSON (seconds, no simulation)
python src/hyperuniformity_analysis.py --n 1024 --seeds 4 --steps 2000 \
       --snaps 1000,2000 --stages 2000 --no-scan --cache f1024.npz \
       --compare data/hyperuniformity_512.json --figure-only
```

The $512^2$ comparison ensemble is reproduced with `--n 512 --seeds 6 --steps 4000`.

**Field caches are not committed.** The snapshot `.npz` files are ~60 MB and are regenerated exactly by step 1 — the solver is deterministic given the seed, and checkpointed runs reproduce uninterrupted trajectories bit for bit.

---

## Why the $1024^2$ ensemble stops at 2000 steps

The $1024^2$ runs stop at 2000 steps while the $512^2$ runs go to 4000. **This is a deliberate choice, not a truncated run.**

The question this paper asks lives at small $k$, and the statistical quality of that band is set by how many domains the box holds. Coarsening reduces the count: continuing to 4000 steps would leave ~1200 domains per configuration instead of ~2736, raising the noise on exactly the wavevectors the argument depends on. Stopping at 2000 steps maximises the accessible range, giving $k_{\min} = 0.0061$ cell$^{-1}$ against a coarsening peak at $0.265$ — about 1.7 decades.

The two ensembles overlap at 1000 and 2000 steps, and the finite-size comparison in the paper (Fig. 3b) is made at the *same* stage in both boxes, which is the only comparison that means anything: the spectrum depends on how far the pattern has coarsened.

---

## Contents

```
src/
  hyperuniformity_analysis.py   simulation, point extraction, S(k), sigma^2(R),
                                nulls, variance budget, journal figures
  mass_autocorrelation.py       mark covariance, mark-crowding correlation,
                                mark-permutation attribution
  cooling_2d.py                 VENDORED phase-field solver (see file header)
data/
  hyperuniformity_1024.json     primary ensemble: all statistics and nulls
  hyperuniformity_512.json      comparison ensemble (3 coarsening stages)
  mass_autocorrelation.json     c_mm(r), crowding correlation, attribution
  domain_marks_stage2000.csv    10,942 domains: seed, x, y, area, solute mass
figures/                        fig_1..fig_4 and fig_mass_autocorrelation (pdf+png)
paper/                          manuscript source (REVTeX 4.2) and figures
```

### `domain_marks_stage2000.csv`

Every quantity in the variance-budget section and in the mass-autocorrelation section can be recomputed from this file alone, without repeating the simulations. Columns: `seed`, `x_cells`, `y_cells` (centroid, grid units, periodic box of side 1024), `area_cells`, `solute_mass` (summed excess composition above the phase threshold).

---

## Notes for anyone re-using the code

Three things in the analysis are easy to get wrong and are handled explicitly here; if you adapt the code, keep them.

1. **Periodic everything.** Domain labelling, centroids (circular means), nearest-neighbour distances and counting windows all wrap. A domain straddling the boundary is one domain, not two at opposite edges.
2. **Fixed-$N$ nulls.** The CSR null uses a fixed number of points, matching the closure constraint the data carry. A Poisson-$N$ null would have a slightly larger number variance for a reason unrelated to physics.
3. **Two constraints that are arithmetic, not results.** Exact mass conservation forces $S_c(k=0)=0$; fixed $N$ in a periodic box drives $\sigma^2 \to 0$ as the window approaches the box. Only $k>0$ is fitted, and windows are capped at $R \le L/6$.

The CSR control is run through the identical pipeline in every analysis and should return $R \approx 1$, $S \approx 1$, slope $\approx 2$. If it does not, the pipeline is wrong and no verdict from it is worth anything.

---

## Related work

- Companion study on kinetic arrest and cooling rates in sphalerite (repository `sphalerite-kinetic-arrest`).
- The point-pattern protocol follows R. Chen, *Pockmark fields are locally ordered but not hyperuniform*, Zenodo (2026), doi:10.5281/zenodo.20854182 — a geomorphic system where local regularity coexists with super-Poissonian long-range behaviour.
- Weighted generalisations of hyperuniformity: S. Torquato *et al.*, Phys. Rev. X **16**, 011042 (2026).

## Citation

See `CITATION.cff`. Please cite the paper rather than this repository where possible.

## Licence

Code: MIT (`LICENSE`). Data files and figures: CC-BY-4.0.
