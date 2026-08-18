# -*- coding: utf-8 -*-
"""
mass_autocorrelation.py  --  Paper 4, the last open question

Sections IV C and III F of the manuscript leave two things unresolved:

  * the residual C of the variance budget mixes two effects -- correlation
    between a domain's mass and the local domain count, and correlation
    between the masses of neighbouring domains -- and the budget does not
    separate them;

  * the count spectrum turns slightly upward at the smallest accessible
    wavevectors, alpha = -0.14 +/- 0.04, and we have no explanation.

Both point at the same missing measurement: the spatial autocorrelation of
domain mass. This script measures it, together with two things that make the
measurement interpretable rather than merely suggestive.

WHAT IS COMPUTED

1. The mark covariance function
       c_mm(r) = < dm_i dm_j >_{|r_ij| in bin} / sigma_m^2 ,
   with dm = m - <m>: the correlation between the masses of two domains a
   distance r apart, normalised to 1 at zero separation. Positive values mean
   large domains sit near large domains.

2. The mark-crowding correlation
       corr( m_i , n_i(r) ) ,
   where n_i(r) counts the neighbours of domain i within r. This is the OTHER
   half of the residual: whether a domain's mass knows about how crowded its
   surroundings are. Together, 1 and 2 separate what the variance budget could
   only lump together.

3. A mark-shuffle attribution. The masses are randomly permuted over the SAME
   positions and the variance budget of Sec. IV is recomputed. Under the
   permutation the marks are independent of position by construction while the
   positions, the mass distribution, and hence CV are all untouched. Two
   outcomes are informative and they are opposite:
       * if the shuffled residual is ~0, the compound-sum identity is behaving
         as derived and the whole of the measured residual comes from
         mass-position coupling;
       * if the shuffled residual is large, the baseline is being violated by
         something other than coupling -- an estimator problem -- and the
         interpretation in Sec. IV would need revisiting.
   This is the check that decides whether Sec. IV is measuring what it claims.

WHY THE NULL IS A PERMUTATION AND NOT AN ANALYTIC BAND

Two constraints bias c_mm(r) in ways that are awkward to write down but exact
under permutation. Total solute is fixed within a configuration, so the marks
sum to a constant and sum_{i != j} dm_i dm_j = -sum_i dm_i^2 is forced: the
average of c_mm over ALL pairs is not zero but about -1/(N-1). And the domains
are not Poisson distributed, so the pairs available in each distance bin are
themselves correlated. Permuting the marks over the observed positions
reproduces both effects exactly, which an analytic confidence band would not.
Anything outside the permutation envelope is therefore a real mark-position
effect and not a constraint artefact.

WHAT THIS CAN AND CANNOT SETTLE

It can separate the two contributions to the residual C. It can show whether
domain mass is organised in space at all.

It cannot, by itself, explain the low-k upturn of the COUNT spectrum. Mass
correlations constrain masses; the count spectrum is a statement about
positions. The bridge between them is the rigidity of the solute -- if the
mass in a region is nearly fixed and the domains there are systematically
large, the count in that region must be low -- and that bridge is a further
inference, not something this measurement establishes on its own. If c_mm(r)
turns out to be flat and inside the null envelope, the upturn remains
unexplained and the honest course is to say so in Sec. V C rather than to
reach for the next available correlate.

Usage
-----
    python mass_autocorrelation.py --cache f1024.npz --stage 2000 --seeds 4

Outputs
-------
    data/mass_autocorrelation.json      curves, nulls, and the attribution
    data/domain_marks_stage<S>.csv      per-domain x, y, area, mass (archivable)
    figures/fig_mass_autocorrelation.{png,pdf}
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from hyperuniformity_analysis import (
    phase_threshold, domain_centroids, variance_budget, _mpl, COL2,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_OUT = os.path.join(_HERE, os.pardir, "figures")
_DATA = os.path.join(_HERE, os.pardir, "data")


# ======================================================================
# Pair machinery
# ======================================================================
def pair_table(pts, L, r_max, n_bins=18, r_min=None):
    """
    Upper-triangle pair separations on the torus, bucketed into log-spaced
    distance bins.

    The bin index is computed once and reused for every mark permutation; with
    ~2700 domains there are ~3.7 million pairs, and recomputing their
    separations 200 times would dominate the runtime for no reason.

    Bins are logarithmic because the interesting range spans from below one
    domain spacing to several, and linear bins would put almost every pair in
    the largest one.
    """
    N = len(pts)
    i, j = np.triu_indices(N, k=1)
    d = pts[i] - pts[j]
    d -= L * np.round(d / L)
    r = np.sqrt((d ** 2).sum(1))
    if r_min is None:
        r_min = max(r.min(), 1e-6)
    edges = np.geomspace(r_min, r_max, n_bins + 1)
    idx = np.digitize(r, edges) - 1
    keep = (idx >= 0) & (idx < n_bins)
    centres = np.sqrt(edges[:-1] * edges[1:])
    counts = np.bincount(idx[keep], minlength=n_bins)
    return i[keep], j[keep], idx[keep], centres, counts


def mark_covariance(marks, i, j, idx, n_bins):
    """
    c_mm(r) = < dm_i dm_j >_r / sigma_m^2 for each distance bin.

    Normalising by the variance rather than by <m>^2 makes the function start
    near 1 for a hypothetical pair at zero separation and makes its magnitude
    directly readable as a correlation coefficient.
    """
    dm = marks - marks.mean()
    var = dm.var()
    if var <= 0:
        return np.full(n_bins, np.nan)
    prod = dm[i] * dm[j]
    tot = np.bincount(idx, weights=prod, minlength=n_bins)
    cnt = np.bincount(idx, minlength=n_bins).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return tot / np.where(cnt > 0, cnt, np.nan) / var


def mark_crowding(pts, marks, L, radii):
    """
    corr(m_i, n_i(r)): does a domain's mass know how crowded its neighbourhood
    is? Neighbours are counted within r on the torus, excluding the domain
    itself.
    """
    N = len(pts)
    out = []
    for r in radii:
        n = np.empty(N)
        step = max(1, int(2e6 // max(N, 1)))
        for a in range(0, N, step):
            b = min(a + step, N)
            d = pts[None, :, :] - pts[a:b, None, :]
            d -= L * np.round(d / L)
            n[a:b] = ((d ** 2).sum(-1) < r * r).sum(1) - 1.0
        out.append(float(np.corrcoef(marks, n)[0, 1]) if n.std() > 0
                   else float("nan"))
    return np.asarray(out)


# ======================================================================
# Driver
# ======================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True,
                    help="npz of snapshot fields written by "
                         "hyperuniformity_analysis.py --simulate-only. The "
                         "results JSON does NOT carry per-domain positions or "
                         "masses, only summary statistics, so the marks are "
                         "re-derived from the fields here.")
    ap.add_argument("--stage", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--bins", type=int, default=18)
    ap.add_argument("--shuffles", type=int, default=200)
    args = ap.parse_args()

    os.makedirs(_OUT, exist_ok=True)
    os.makedirs(_DATA, exist_ok=True)
    z = np.load(args.cache)
    rng = np.random.default_rng(20260817)

    cmm_all, cmm_null_lo, cmm_null_hi, crowd_all, crowd_null = [], [], [], [], []
    budget_real, budget_shuf = [], []
    csv_rows = []
    centres = radii = None

    for s in range(args.seeds):
        c = z[f"s{args.stage}_{s}"]
        L = float(c.shape[0])
        thr = phase_threshold(c)
        pts, areas, marks = domain_centroids(c, thr)
        N = len(pts)
        rho = N / L ** 2
        d_typ = 1.0 / np.sqrt(rho)
        print(f"  seed {s}: {N} domains, spacing {d_typ:.1f} cells, "
              f"mass CV {marks.std() / marks.mean():.3f}")

        for (y, x), a, m in zip(pts, areas, marks):
            csv_rows.append((s, x, y, a, m))

        # --- 1. mark covariance, with the permutation envelope ---
        i, j, idx, centres_s, cnt = pair_table(pts, L, r_max=L / 4.0,
                                               n_bins=args.bins,
                                               r_min=0.4 * d_typ)
        centres = centres_s
        cmm_all.append(mark_covariance(marks, i, j, idx, args.bins))
        null = np.empty((args.shuffles, args.bins))
        for t in range(args.shuffles):
            null[t] = mark_covariance(rng.permutation(marks), i, j, idx,
                                      args.bins)
        cmm_null_lo.append(np.percentile(null, 2.5, axis=0))
        cmm_null_hi.append(np.percentile(null, 97.5, axis=0))

        # --- 2. mark-crowding correlation ---
        radii = np.geomspace(0.8 * d_typ, L / 8.0, 10)
        crowd_all.append(mark_crowding(pts, marks, L, radii))
        crowd_null.append(np.mean([mark_crowding(pts, rng.permutation(marks),
                                                 L, radii)
                                   for _ in range(3)], axis=0))

        # --- 3. mark-shuffle attribution of the variance budget ---
        w_radii = np.geomspace(0.5 * d_typ, L / 6.0, 12)
        budget_real.append(variance_budget(pts, marks, L, w_radii))
        budget_shuf.append(variance_budget(pts, rng.permutation(marks), L,
                                           w_radii))

    cmm = np.nanmean(cmm_all, axis=0)
    lo = np.nanmean(cmm_null_lo, axis=0)
    hi = np.nanmean(cmm_null_hi, axis=0)
    crowd = np.nanmean(crowd_all, axis=0)
    crowd0 = np.nanmean(crowd_null, axis=0)

    def _mean_budget(bs):
        return [{k: float(np.mean([b[i][k] for b in bs])) for k in bs[0][i]}
                for i in range(len(bs[0]))]

    br, bs_ = _mean_budget(budget_real), _mean_budget(budget_shuf)

    print("\n  mark covariance c_mm(r)  (outside the null band = real)")
    print("      r (cells)    c_mm      null 2.5%   null 97.5%   verdict")
    for r, v, a, b in zip(centres, cmm, lo, hi):
        flag = "OUTSIDE" if (v < a or v > b) else "inside"
        print(f"      {r:8.1f} {v:+9.4f} {a:+11.4f} {b:+12.4f}   {flag}")

    print("\n  mark-crowding corr(m_i, n_i(r))")
    for r, v, v0 in zip(radii, crowd, crowd0):
        print(f"      r = {r:7.1f}:  {v:+.4f}   (shuffled: {v0:+.4f})")

    print("\n  variance-budget attribution: residual C with real vs shuffled marks")
    print("      R        C(real)    C(shuffled)   ratio measured/baseline")
    for a, b in zip(br, bs_):
        print(f"      {a['R']:7.1f} {a['residual']:10.2f} {b['residual']:12.2f}"
              f"     {a['ratio']:.3f} vs {b['ratio']:.3f}")

    out = dict(stage=args.stage, seeds=args.seeds, shuffles=args.shuffles,
               r=centres.tolist(), c_mm=cmm.tolist(),
               c_mm_null_lo=lo.tolist(), c_mm_null_hi=hi.tolist(),
               crowding_r=radii.tolist(), crowding=crowd.tolist(),
               crowding_shuffled=crowd0.tolist(),
               budget_real=br, budget_shuffled=bs_)
    with open(os.path.join(_DATA, "mass_autocorrelation.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    csv_path = os.path.join(_DATA, f"domain_marks_stage{args.stage}.csv")
    with open(csv_path, "w") as fh:
        fh.write("seed,x_cells,y_cells,area_cells,solute_mass\n")
        for s, x, y, a, m in csv_rows:
            fh.write(f"{s},{x:.4f},{y:.4f},{a:.0f},{m:.6f}\n")

    make_figure(out, os.path.join(_OUT, "fig_mass_autocorrelation"))
    print(f"\n  data    -> {os.path.join(_DATA, 'mass_autocorrelation.json')}")
    print(f"  marks   -> {csv_path}")
    print(f"  figure  -> {os.path.join(_OUT, 'fig_mass_autocorrelation.png')}")


def make_figure(out, outfile):
    plt = _mpl()
    plt.rcParams.update({"font.size": 7.5, "axes.labelsize": 8,
                         "legend.fontsize": 6.5, "axes.titlesize": 8})
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL2, COL2 * 0.36))

    r = np.asarray(out["r"])
    a.fill_between(r, out["c_mm_null_lo"], out["c_mm_null_hi"], color="0.85",
                   label="mark-permutation null (95%)")
    a.semilogx(r, out["c_mm"], "-", color="#c1272d", marker="o", ms=3, lw=1.2,
               label=r"$c_{mm}(r)$")
    a.axhline(0.0, color="0.5", lw=0.6, ls=":")
    a.set_xlabel(r"separation $r$ (cells)")
    a.set_ylabel(r"$\langle \delta m_i \delta m_j\rangle_r / \sigma_m^2$")
    a.legend(frameon=False, loc="upper right")
    a.set_title("(a) do neighbouring domains have similar masses?", loc="left")

    rc = np.asarray(out["crowding_r"])
    b.semilogx(rc, out["crowding"], "-", color="#1b7837", marker="s", ms=3,
               lw=1.2, label=r"corr$(m_i, n_i(r))$")
    b.semilogx(rc, out["crowding_shuffled"], "--", color="0.5", marker="x",
               ms=3, lw=1.0, label="marks permuted")
    b.axhline(0.0, color="0.5", lw=0.6, ls=":")
    b.set_xlabel(r"neighbourhood radius $r$ (cells)")
    b.set_ylabel("correlation")
    b.legend(frameon=False, loc="upper right")
    b.set_title("(b) does a domain's mass know its crowding?", loc="left")

    fig.tight_layout(pad=0.4)
    for ext in ("pdf", "png"):
        fig.savefig(outfile + "." + ext, bbox_inches="tight", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
