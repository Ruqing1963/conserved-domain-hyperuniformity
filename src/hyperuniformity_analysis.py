# -*- coding: utf-8 -*-
"""
hyperuniformity_analysis.py  --  Paper 3, companion analysis

Is the spatial arrangement of the indium-rich nanodomains produced by the
phase-field model DISORDERED HYPERUNIFORM?

    A point pattern is hyperuniform if its long-wavelength density fluctuations
    vanish: S(k) -> 0 as k -> 0, equivalently the number variance in a window of
    radius R grows like the window PERIMETER (sigma^2 ~ R in 2D) rather than
    like its AREA (sigma^2 ~ R^2, the Poisson value).

The protocol is the one Chen (2026) applied to Mediterranean seabed pockmarks:
a LOCAL diagnostic (the Clark-Evans nearest-neighbour index) and two LONG-RANGE
diagnostics (the radially averaged structure factor and the number variance),
each read against complete-spatial-randomness and hard-core nulls. There the
verdict was negative -- pockmark fields are locally ordered but long-range
super-Poissonian. Running the same protocol on a system whose dynamics conserve
mass LOCALLY is the controlled counter-experiment.

WHY THE ANSWER IS NOT KNOWN IN ADVANCE, AND WHAT IS ACTUALLY MEASURED

The Cahn-Hilliard equation is a continuity equation, dc/dt = -div J. Composition
is not merely conserved in total, it can only move by flowing, so a density
excess at wavelength 2pi/k can only be built by transporting solute across that
distance, which takes a time ~ k^-4. Long-wavelength composition fluctuations
are therefore frozen at their initial (small) amplitude while short-wavelength
ones grow, and the CONCENTRATION FIELD acquires a suppressed low-k structure
factor. That much is expected.

The object this script is really about is different: the POINT PATTERN of domain
centroids. It is not the field. It throws away domain size, and mass
conservation constrains the field, not the count of domains -- a region can hold
its share of solute in few large domains or in many small ones. Whether the
centroids inherit hyperuniformity from the field is an empirical question, and
polydispersity is the obvious way for the inheritance to fail. Both objects are
therefore measured, separately, and reported separately.

Three further things are measured rather than assumed:

  * the DIGITISATION. Centroids come from thresholding a field. If the verdict
    moved with the threshold it would be an artefact, so the threshold is
    scanned and the scan is reported.

  * the ESTIMATORS. Every statistic is also run on a complete-spatial-random
    point set in the same periodic box at the same N. That control must return
    R = 1, S(k) = 1 and a number-variance slope of 2. If it does not, the
    pipeline is wrong and no verdict from it means anything.

  * the FINITE BOX. The smallest accessible wavevector is 2pi/L. With the
    domain spacing this leaves rather less than two decades in k, so the
    measurable claim is the SCALING over the accessible range, not a precise
    hyperuniformity class.

TWO CONSTRAINTS THAT MUST NOT BE MISREAD AS RESULTS

  1. The solver conserves total composition exactly and the box is periodic, so
     the field has S(k = 0) = 0 identically. That is arithmetic, not physics.
     Only k > 0 is ever fitted or plotted.

  2. A configuration with N points fixed in a periodic box has a number
     variance that is driven to zero as the window approaches the whole box,
     for the same trivial reason. Windows are therefore restricted to
     R <= L/6, and the CSR null is generated at FIXED N so that it carries the
     identical constraint and the comparison stays honest.

Usage
-----
    python hyperuniformity_analysis.py                 # 512^2, 4 seeds
    python hyperuniformity_analysis.py --quick         # 256^2, 2 seeds
    python hyperuniformity_analysis.py --n 768 --seeds 8

Outputs
-------
    data/hyperuniformity.json          all numbers, including the nulls
    figures/fig_hyperuniformity.{png,pdf}
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from scipy import ndimage

from cooling_2d import SphaleriteFreeEnergy, CahnHilliard2D

_HERE = os.path.dirname(os.path.abspath(__file__))
_OUT = os.path.join(_HERE, os.pardir, "figures")
_DATA = os.path.join(_HERE, os.pardir, "data")

# Published pockmark values (Chen 2026, Table 2), for the order plane only.
POCKMARKS = {
    "Offshore Zannone": (1.08, 2.54),
    "NW Calabrian margin": (0.47, 19.4),
    "Malta Plateau": (1.26, 5.36),
}


# ======================================================================
# 1. From field to point set
# ======================================================================
def label_periodic(mask):
    """
    Connected components of a boolean mask on a TORUS.

    scipy.ndimage.label treats the array edges as boundaries, which would split
    every domain that straddles the periodic wrap into two and place a spurious
    pair of centroids at opposite sides of the box. Components touching
    opposite edges are therefore merged by union-find afterwards.
    """
    lab, n = ndimage.label(mask)
    parent = np.arange(n + 1)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for A, B in ((lab[0, :], lab[-1, :]), (lab[:, 0], lab[:, -1])):
        sel = (A > 0) & (B > 0)
        for a, b in zip(A[sel], B[sel]):
            union(int(a), int(b))

    root = np.array([find(i) for i in range(n + 1)])
    keep = np.unique(root[1:]) if n else np.array([], dtype=int)
    lut = np.zeros(n + 1, dtype=int)
    for new, old in enumerate(keep, start=1):
        lut[root == old] = new
    lut[0] = 0
    return lut[lab], len(keep)


def phase_threshold(c, lo=2.0, hi=98.0):
    """
    Midpoint between the two phase compositions.

    Taken as the midpoint of robust percentiles rather than of (min, max): the
    extremes of a spectral solution carry ringing, and rather than of the mean,
    which for an off-critical composition sits inside the solute-poor phase and
    would cut the domains at their skirts.
    """
    return float(0.5 * (np.percentile(c, lo) + np.percentile(c, hi)))


def domain_centroids(c, thr, min_cells=8):
    """
    Mass-weighted centroids of the solute-rich domains, in grid units, on a
    torus.

    The weight is the excess composition (c - thr) above the threshold, so the
    centroid is the centre of MASS of the domain's solute rather than the
    centre of its silhouette. On a torus a plain arithmetic mean of coordinates
    is wrong for any domain crossing the wrap -- it returns the middle of the
    box -- so each coordinate is averaged as a circular mean.

    Domains smaller than min_cells are dropped: below a few cells a "domain" is
    an unresolved composition ripple, and admitting them adds points whose
    positions are noise.

    Returns the centroids, the domain areas in cells, and the domain solute
    masses (the summed excess composition). The masses matter: they are what
    the dynamics conserve, and the mass-weighted spectrum is the diagnostic
    that separates "the domains are placed uniformly" from "the SOLUTE is
    placed uniformly", which are not the same statement.
    """
    n = c.shape[0]
    lab, ndom = label_periodic(c > thr)
    if ndom == 0:
        return np.zeros((0, 2)), np.zeros(0), np.zeros(0)

    w = np.clip(c - thr, 0.0, None)
    ang = 2 * np.pi * np.arange(n) / n
    sin_a, cos_a = np.sin(ang), np.cos(ang)

    areas = np.bincount(lab.ravel(), minlength=ndom + 1)[1:]
    pts, keep_area, keep_mass = [], [], []
    rows, cols = np.nonzero(lab)
    order = np.argsort(lab[rows, cols], kind="stable")
    rows, cols = rows[order], cols[order]
    starts = np.cumsum(np.concatenate(([0], areas)))
    for i in range(ndom):
        if areas[i] < min_cells:
            continue
        sl = slice(starts[i], starts[i + 1])
        r, cc = rows[sl], cols[sl]
        ww = w[r, cc]
        if ww.sum() <= 0:
            continue
        xy = []
        for coord in (r, cc):
            th = np.arctan2((ww * sin_a[coord]).sum(),
                            (ww * cos_a[coord]).sum()) % (2 * np.pi)
            xy.append(th * n / (2 * np.pi))
        pts.append(xy)
        keep_area.append(areas[i])
        keep_mass.append(float(ww.sum()))
    return (np.asarray(pts, float).reshape(-1, 2),
            np.asarray(keep_area, float), np.asarray(keep_mass, float))


# ======================================================================
# 2. Local order
# ======================================================================
def clark_evans(pts, L):
    """
    Clark-Evans nearest-neighbour index on a torus.

    R = <d_NN> / (1 / (2 sqrt(rho))); R < 1 clustered, 1 random, > 1 regular.

    The usual difficulty with this statistic -- that points near the boundary
    have their true nearest neighbour outside the window, biasing <d_NN> upward
    and faking regularity -- does not arise here. The simulation box IS the
    torus, so the periodic minimum-image distance is exact and no edge
    correction is applied or needed.
    """
    N = len(pts)
    if N < 2:
        return dict(N=N, R=float("nan"), z=float("nan"), d_nn=float("nan"))
    d = pts[:, None, :] - pts[None, :, :]
    d -= L * np.round(d / L)
    dist = np.sqrt((d ** 2).sum(-1))
    np.fill_diagonal(dist, np.inf)
    dnn = dist.min(1)
    rho = N / L ** 2
    expected = 1.0 / (2.0 * np.sqrt(rho))
    sigma = np.sqrt((4.0 - np.pi) / (4.0 * np.pi * rho * N))
    return dict(N=int(N), R=float(dnn.mean() / expected),
                z=float((dnn.mean() - expected) / sigma),
                d_nn=float(np.median(dnn)), d_nn_min=float(dnn.min()))


# ======================================================================
# 3. Long-range order
# ======================================================================
def structure_factor_points(pts, L, n_modes=48, weights=None):
    """
    S(k) = |sum_j exp(-i k.r_j)|^2 / N on the exact reciprocal lattice of the
    periodic box, k = 2 pi (nx, ny) / L.

    Because the box is periodic these wavevectors are the natural modes of the
    system: no window function, no taper, no edge correction, and hence none of
    the low-k leakage that makes S(k) hard to interpret for a bounded field
    survey. For an uncorrelated set of N points on this lattice, E[S(k)] = 1 for
    every k != 0, which is what makes the CSR control a sharp test.

    k = 0 is excluded: it equals N by construction and says only that the
    points were counted.

    With `weights` supplied the same expression is evaluated for the weighted
    density, S_w(k) = |sum_j w_j exp(-i k.r_j)|^2 / sum_j w_j^2, normalised so
    that uncorrelated positions again give 1 whatever the weights. Passing the
    domain solute masses turns this into a spectrum of the SOLUTE
    distribution -- a coarse-grained version of the field's own S(k) -- while
    the unweighted version is a spectrum of domain COUNTS. Comparing the two is
    how the analysis distinguishes conservation of mass from organisation of
    domains.
    """
    N = len(pts)
    ns = np.arange(-n_modes, n_modes + 1)
    kv = 2 * np.pi * ns / L
    w = np.ones(N) if weights is None else np.asarray(weights, float)
    norm = (w ** 2).sum()
    Ax = np.exp(-1j * np.outer(pts[:, 0], kv)) * w[:, None]   # (N, nk)
    Ay = np.exp(-1j * np.outer(pts[:, 1], kv))
    rho_k = Ax.T @ Ay                                         # (nk, nk)
    S = (np.abs(rho_k) ** 2 / norm).ravel()
    KX, KY = np.meshgrid(kv, kv, indexing="ij")
    k = np.sqrt(KX ** 2 + KY ** 2).ravel()
    sel = k > 0
    return k[sel], S[sel]


def structure_factor_field(c, dx=1.0):
    """
    S(k) of the concentration field itself, |c_k|^2 / N_cells.

    Normalised so that a spatially uncorrelated field would give a flat S(k).
    k = 0 is dropped: the solver conserves the mean composition exactly, so
    that mode is identically zero and carries no information.
    """
    n = c.shape[0]
    d = c - c.mean()
    S = (np.abs(np.fft.fft2(d)) ** 2 / (n * n)).ravel()
    kv = 2 * np.pi * np.fft.fftfreq(n, d=dx)
    KX, KY = np.meshgrid(kv, kv, indexing="ij")
    k = np.sqrt(KX ** 2 + KY ** 2).ravel()
    sel = k > 0
    return k[sel], S[sel]


def radial_average(k, S, n_bins=44, k_max=None):
    """
    Bin (k, S) pairs, pooled over configurations, into |k| shells.

    The bins are LOGARITHMIC. Linear bins put almost every shell above the
    coarsening peak and leave three or four points below it, which is the half
    of the axis the hyperuniformity question lives on; a power law also has to
    be fitted on a logarithmic abscissa if the fit is not to be dominated by
    the largest wavevectors.
    """
    sel = k > 0
    k, S = k[sel], S[sel]
    if k_max is not None:
        sel = k <= k_max
        k, S = k[sel], S[sel]
    if k.size == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    edges = np.geomspace(k.min() * 0.999, k.max() * 1.001, n_bins + 1)
    idx = np.digitize(k, edges) - 1
    kk, SS, cnt = [], [], []
    for i in range(n_bins):
        m = idx == i
        if m.sum() >= 1:
            kk.append(k[m].mean())
            SS.append(S[m].mean())
            cnt.append(int(m.sum()))
    return np.asarray(kk), np.asarray(SS), np.asarray(cnt)


def low_k_exponent(k, S, k_lo, k_hi):
    """
    Fit S ~ k^alpha over [k_lo, k_hi] and report alpha with its standard error.

    alpha > 0 with S falling to zero is hyperuniform; alpha = 0 (flat, S = 1) is
    Poisson; alpha < 0 (rising) is the super-Poissonian behaviour of a clustered
    or heterogeneous pattern. The classes of Torquato (2018) are alpha > 1
    (class I), alpha = 1 (class II) and 0 < alpha < 1 (class III), but
    separating them needs a k range this box does not have, so alpha is reported
    with its uncertainty and the class is not claimed.
    """
    sel = (k >= k_lo) & (k <= k_hi) & (S > 0)
    if sel.sum() < 4:
        return float("nan"), float("nan")
    x, y = np.log(k[sel]), np.log(S[sel])
    A = np.vstack([x, np.ones_like(x)]).T
    coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    dof = max(sel.sum() - 2, 1)
    s2 = (res[0] / dof) if res.size else 0.0
    cov = s2 * np.linalg.inv(A.T @ A)
    return float(coef[0]), float(np.sqrt(cov[0, 0]))


def number_variance(pts, L, radii, n_windows=None, rng=None):
    """
    Variance of the point count in disks of radius R placed at random in the
    torus.

    Disks wrap, so every window is statistically equivalent and there is no
    edge-effect correction. R is kept small compared with the box (see the
    module docstring): with N fixed, a window approaching the box size has a
    count approaching N and a variance approaching zero, which would masquerade
    as perfect hyperuniformity.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    # The variance of the estimate falls with the number of windows AND with
    # the number of points each window sees, so a dense pattern needs fewer
    # windows for the same precision. Scaling keeps a 4000-point configuration
    # from costing four times what a 1000-point one does for no gain.
    if n_windows is None:
        n_windows = int(np.clip(3.0e6 / max(len(pts), 1), 1200, 3000))
    out = []
    for R in radii:
        ctr = rng.random((n_windows, 2)) * L
        counts = np.empty(n_windows, dtype=np.int32)
        step = max(1, int(2e6 // max(len(pts), 1)))
        for a in range(0, n_windows, step):
            b = min(a + step, n_windows)
            d = pts[None, :, :] - ctr[a:b, None, :]
            d -= L * np.round(d / L)
            counts[a:b] = ((d ** 2).sum(-1) < R * R).sum(1)
        out.append(counts.var())
    return np.asarray(out, float)


def count_vs_mass_variance(pts, masses, L, radii, n_windows=None, rng=None):
    """
    In the same windows, the variance of the DOMAIN COUNT and the variance of
    the SOLUTE MASS. This is the compensation hypothesis measured directly.

    The claim to be tested is that a region is free to hold its solute budget
    in few large domains or in many small ones, so that number fluctuations are
    cancelled by anti-correlated size fluctuations and only the count spectrum
    keeps a plateau. Written in windows, the claim is that

        Var[M(R)] / <m>^2   <<   Var[N(R)]

    where <m> is the mean domain mass: expressing the mass variance in units of
    a typical domain converts it into the number of domains' worth of solute
    that the window's content fluctuates by, which is directly comparable with
    the number variance. If the two curves coincide, size carries no
    information and mass fluctuation is simply count fluctuation. If the mass
    curve falls below, size is absorbing part of it.

    This version deletes nothing. The filtered-subset test has to remove
    points, and mass turns out to be spatially correlated, so removing points
    by mass also removes them by position; here every domain is used and that
    confound cannot arise.
    """
    rng = np.random.default_rng(7) if rng is None else rng
    if n_windows is None:
        n_windows = int(np.clip(3.0e6 / max(len(pts), 1), 1200, 3000))
    m = np.asarray(masses, float)
    m_bar = m.mean()
    var_n, var_m, corr = [], [], []
    for R in radii:
        ctr = rng.random((n_windows, 2)) * L
        N_w = np.empty(n_windows)
        M_w = np.empty(n_windows)
        step = max(1, int(2e6 // max(len(pts), 1)))
        for a in range(0, n_windows, step):
            b = min(a + step, n_windows)
            d = pts[None, :, :] - ctr[a:b, None, :]
            d -= L * np.round(d / L)
            inside = (d ** 2).sum(-1) < R * R
            N_w[a:b] = inside.sum(1)
            M_w[a:b] = inside @ m
        var_n.append(float(N_w.var()))
        var_m.append(float(M_w.var() / m_bar ** 2))
        corr.append(float(np.corrcoef(N_w, M_w)[0, 1]))
    return np.asarray(var_n), np.asarray(var_m), np.asarray(corr)


def variance_budget(pts, masses, L, radii, n_windows=None, rng=None):
    """
    Decompose the solute-mass variance in a window into what independent
    domains would give and what is left over.

    For a sum of N masses drawn independently of one another and of N, the
    variance of a compound random sum is exactly

        Var[M] = <N> Var[m] + <m>^2 Var[N],

    so in units of a typical domain,

        Var[M] / <m>^2 = <N> CV^2 + Var[N],                            (baseline)

    with CV the coefficient of variation of the domain masses. Both terms on
    the right are measured -- <N> and Var[N] from the same windows, CV from the
    mass distribution -- so the baseline is a prediction with no free
    parameter, and the residual

        C = Var[M]/<m>^2 - <N> CV^2 - Var[N]

    is the part of the solute rigidity that independence cannot account for. A
    large negative C means the system is actively cancelling the fluctuation
    that polydispersity alone would impose: where there are extra domains, they
    are smaller.

    WHAT C IS AND IS NOT. C lumps together two effects that this measurement
    does not separate: correlation between a domain's mass and the local domain
    count, and correlation between the masses of neighbouring domains. Both are
    forms of the same compensation, but the decomposition should not be read as
    isolating one of them. Cov(N, mean mass in window) is reported alongside
    because it exhibits the count-size part directly.

    WHY NOT A SYNTHETIC MODEL. The obvious alternative -- build a hyperuniform
    mass field, cut it into domains with the observed mass distribution, and
    measure the resulting count spectrum -- requires choosing where to put the
    domains, and that choice IS the count spectrum under test. The budget above
    assumes nothing about positions: it is an identity applied to measured
    quantities.
    """
    rng = np.random.default_rng(11) if rng is None else rng
    if n_windows is None:
        n_windows = int(np.clip(3.0e6 / max(len(pts), 1), 1200, 3000))
    m = np.asarray(masses, float)
    m_bar = m.mean()
    cv2 = float((m.std() / m_bar) ** 2)
    rows = []
    for R in radii:
        ctr = rng.random((n_windows, 2)) * L
        N_w = np.empty(n_windows)
        M_w = np.empty(n_windows)
        step = max(1, int(2e6 // max(len(pts), 1)))
        for a in range(0, n_windows, step):
            b = min(a + step, n_windows)
            d = pts[None, :, :] - ctr[a:b, None, :]
            d -= L * np.round(d / L)
            inside = (d ** 2).sum(-1) < R * R
            N_w[a:b] = inside.sum(1)
            M_w[a:b] = inside @ m
        varM = float(M_w.var() / m_bar ** 2)
        varN = float(N_w.var())
        meanN = float(N_w.mean())
        base = meanN * cv2 + varN
        ok = N_w > 0
        mean_size = M_w[ok] / (N_w[ok] * m_bar)
        cov_n_size = float(np.cov(N_w[ok], mean_size)[0, 1])
        corr_n_size = float(np.corrcoef(N_w[ok], mean_size)[0, 1])
        sd_size = float(mean_size.std())
        rows.append(dict(R=float(R), meanN=meanN, varN=varN, varM=varM,
                         cv2=cv2, term_size=meanN * cv2, baseline=float(base),
                         residual=float(varM - base),
                         ratio=float(varM / base) if base > 0 else float("nan"),
                         cov_count_size=cov_n_size,
                         corr_count_size=corr_n_size, sd_mean_size=sd_size))
    return rows


# ======================================================================
# 4. Null models
# ======================================================================
def null_csr(N, L, rng):
    """Binomial point process: N independent uniform points in the torus.

    Fixed N, not Poisson N, so that it carries the same closure constraint as a
    simulated configuration -- otherwise the null would have a slightly LARGER
    number variance than the data for a reason having nothing to do with
    physics.
    """
    return rng.random((N, 2)) * L


def null_rsa(N, L, d_exclusion, rng, max_tries=400):
    """
    Random sequential adsorption of N hard discs of diameter d_exclusion.

    This is the steric null: the strongest local regularity obtainable from
    exclusion alone, with no long-range organisation. It is the reference that
    separates "the domains cannot overlap" from "the domains are arranged".
    RSA is emphatically NOT hyperuniform, and its S(k -> 0) sits somewhat below
    1 -- a useful reminder that a value below unity is not by itself a
    hyperuniform signal.

    If the requested coverage exceeds the 2D RSA saturation limit (~0.547 of the
    area) the exclusion distance is reduced until the target is attainable, and
    the value actually used is returned.
    """
    d = float(d_exclusion)
    phi_max = 0.52
    while N * np.pi * (d / 2) ** 2 / L ** 2 > phi_max and d > 1e-6:
        d *= 0.95
    pts = np.empty((N, 2))
    m = 0
    tries = 0
    while m < N and tries < max_tries * N:
        p = rng.random(2) * L
        if m == 0:
            pts[0] = p
            m = 1
            continue
        dd = pts[:m] - p
        dd -= L * np.round(dd / L)
        if (dd ** 2).sum(1).min() >= d * d:
            pts[m] = p
            m += 1
        tries += 1
    return pts[:m], d


def null_perturbed_lattice(N, L, sigma_frac, rng):
    """
    Square lattice with independent Gaussian displacements: the standard
    disordered-hyperuniform reference. Perturbation destroys the Bragg peaks but
    not the suppression of long-wavelength fluctuations, so this pattern shows
    what the diagnostics look like when the answer is unambiguously yes.
    """
    m = int(round(np.sqrt(N)))
    a = L / m
    g = (np.arange(m) + 0.5) * a
    X, Y = np.meshgrid(g, g, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    pts = pts + rng.normal(0.0, sigma_frac * a, pts.shape)
    return pts % L


def polydispersity_test(sets, masses, L, retentions=(0.30, 0.50, 0.70),
                        n_modes=128, rng=None, n_low=8):
    """
    Does the low-k plateau of the COUNT spectrum come from domain
    polydispersity?

    The hypothesis under test is that mass conservation fixes the local solute
    budget while leaving free how many domains hold it, so that a region can
    trade one large domain for two small ones without violating anything. If
    that is the mechanism, then restricting the analysis to domains of nearly
    equal mass -- a monodisperse subset, in which the trade is not available --
    should lower the plateau.

    THE CONTROL THAT MAKES THIS A TEST RATHER THAN AN ARTEFACT

    Selecting a subset removes points, and removing points at random by itself
    drives S(k) towards the Poisson value: for independent thinning with
    retention probability p,

        S_thinned(k) = 1 + p [ S_full(k) - 1 ],

    so a pattern with S(0) = 0.13 thinned to 40 % of its points returns
    S(0) = 0.65 with no change whatever in the underlying order. Comparing the
    mass-filtered subset with the FULL pattern would therefore show the plateau
    RISING and prove nothing. Every filtered subset is accordingly matched
    against a random subset of exactly the same size drawn from the same
    configuration, and it is the difference between those two -- not the
    difference from the full set -- that carries the evidence.

    A filtered plateau below the thinning null means the discarded domains were
    carrying more than their share of the long-wavelength fluctuation, which is
    what the polydispersity account predicts.

    The subset is specified by a RETENTION FRACTION -- keep the q of domains
    whose mass is closest to the median -- rather than by a mass tolerance in
    per cent. A fixed tolerance retains whatever fraction the mass distribution
    happens to put inside it, which varies between configurations and
    coarsening stages and so makes the thinning null a moving target; fixing q
    instead keeps filtered and thinned sets exactly the same size everywhere
    and puts as many points as possible into the comparison, which is where the
    statistical power of this test is.
    """
    rng = np.random.default_rng(2718) if rng is None else rng
    out = []
    for q in retentions:
        S_f, S_r, kept, dropped_cv, cv_all = [], [], [], [], []
        for pts, m in zip(sets, masses):
            if len(pts) < 40:
                continue
            med = np.median(m)
            n_keep = max(30, int(round(q * len(pts))))
            sel = np.zeros(len(m), bool)
            sel[np.argsort(np.abs(m - med))[:n_keep]] = True
            cv_all.append(float(np.std(m) / np.mean(m)))
            k_f, s_f = structure_factor_points(pts[sel], L, n_modes=n_modes)
            idx = rng.choice(len(pts), size=n_keep, replace=False)
            k_r, s_r = structure_factor_points(pts[idx], L, n_modes=n_modes)
            S_f.append((k_f, s_f, clark_evans(pts[sel], L)["R"]))
            S_r.append((k_r, s_r, clark_evans(pts[idx], L)["R"]))
            kept.append(n_keep / len(pts))
            dropped_cv.append(float(np.std(m[sel]) / np.mean(m[sel])))
        if not S_f:
            continue
        rec = dict(retention=float(q), retained=float(np.mean(kept)),
                   cv_kept=float(np.mean(dropped_cv)),
                   cv_all=float(np.mean(cv_all)))
        for tag, pool in (("filtered", S_f), ("thinned", S_r)):
            k = np.concatenate([a for a, _, _ in pool])
            S = np.concatenate([b for _, b, _ in pool])
            rec[tag + "_R"] = float(np.mean([r for _, _, r in pool]))
            kk, SS, _ = radial_average(k, S, n_bins=36)
            rec[tag + "_S_low"] = float(np.mean(SS[:n_low]))
            N_eff = np.mean([len(p) for p in sets]) * np.mean(kept)
            kp = 2 * np.pi * np.sqrt(N_eff) / L
            a, ase = low_k_exponent(kk, SS, 0.9 * 2 * np.pi / L, 0.45 * kp)
            rec[tag + "_alpha"] = float(a)
            rec[tag + "_alpha_se"] = float(ase)
            rec[tag + "_k"] = kk.tolist()
            rec[tag + "_S"] = SS.tolist()
        out.append(rec)
    return out


# ======================================================================
# 5. One coarsening run
# ======================================================================
class CahnHilliard2DFast(CahnHilliard2D):
    """
    The solver of cooling_2d, with scipy's pocketfft in place of numpy's.

    Identical arithmetic -- the same semi-implicit update, the same float64
    precision, the same wavevectors -- and only the transform library differs.
    At 1024^2 on one core that is worth a factor of about 1.7, which is the
    difference between a run that finishes and one that does not. The
    equivalence is asserted numerically at start-up rather than assumed, by
    stepping both versions from the same initial condition and comparing
    (verify_fft_backend below); if the two ever disagree beyond round-off the
    run stops, because a faster answer that is a different answer is worthless.
    """

    def step(self, dt, M):
        from scipy import fft as sfft
        c_hat = sfft.rfft2(self.c, workers=-1)
        mu_hat = sfft.rfft2(self.free.dfdc(self.c), workers=-1)
        self.c = sfft.irfft2(
            (c_hat - dt * M * self.k2 * mu_hat)
            / (1.0 + dt * M * self.kappa * self.k4),
            s=(self.n, self.n), workers=-1)
        return self.c


def verify_fft_backend(free, kappa, dt, n=128, steps=25, tol=1e-10):
    """Step the two backends side by side and return the largest divergence."""
    a = CahnHilliard2D(n, 1.0, kappa, free, seed=99)
    b = CahnHilliard2DFast(n, 1.0, kappa, free, seed=99)
    a.initialise(0.30, noise=0.002)
    b.initialise(0.30, noise=0.002)
    for _ in range(steps):
        a.step(dt, 1.0)
        b.step(dt, 1.0)
    d = float(np.max(np.abs(a.c - b.c)))
    if d > tol:
        raise RuntimeError(f"FFT backends disagree by {d:.2e} after {steps} "
                           "steps; refusing to use the fast path")
    return d


def _solver_setup(n, c0, seed, lam_target=12.0, f_scale=100.0, target_amp=1.02,
                  fast=True):
    """Free energy, kappa, dt and a solver -- exactly as in cooling_2d."""
    free = SphaleriteFreeEnergy(scale=f_scale)
    d2 = free.d2fdc2(c0)
    if d2 >= 0:
        raise ValueError(f"c0 = {c0} is not inside the unstable region")
    kappa = abs(d2) * lam_target ** 2 / (8 * np.pi ** 2)
    kk = np.linspace(1e-3, np.pi, 800)
    k_grow = kk[np.argmax(-kk ** 2 * (d2 + kappa * kk ** 2))]
    dt = (target_amp - 1.0) / (-k_grow ** 2 * d2
                               - target_amp * kappa * k_grow ** 4)
    cls = CahnHilliard2DFast if fast else CahnHilliard2D
    return free, kappa, dt, cls(n, 1.0, kappa, free, seed=seed)


def simulate_chunked(n, c0, seed, steps, snap_steps, ckpt, chunk, fast=True):
    """
    Advance one run by at most `chunk` steps, then checkpoint and return.

    A 1024^2 run to 4000 steps is several times longer than the wall-clock
    budget of the environment this was developed in, and losing it at step 3900
    would mean starting again. The state of a Cahn-Hilliard run is just the
    field, so a checkpoint is cheap and exact: the run is resumed from the
    saved field with the same dt and reproduces the uninterrupted trajectory
    bit for bit. Snapshots already passed are carried in the checkpoint.

    Returns (steps_done, finished).
    """
    free, kappa, dt, sim = _solver_setup(n, c0, seed, fast=fast)
    snaps = {}
    if os.path.exists(ckpt):
        z = np.load(ckpt)
        sim.c = z["c"]
        done = int(z["step"])
        for s in snap_steps:
            if f"snap{s}" in z.files:
                snaps[s] = z[f"snap{s}"]
    else:
        sim.initialise(c0, noise=0.002)
        done = 0
    if done >= steps:
        return done, True

    t0 = time.time()
    target = min(steps, done + chunk)
    for s in range(done + 1, target + 1):
        sim.step(dt, 1.0)
        if s in snap_steps:
            snaps[s] = sim.c.copy()
        if not np.isfinite(sim.c).all():
            raise RuntimeError(f"field diverged at step {s}")
    np.savez_compressed(ckpt, c=sim.c, step=target,
                        **{f"snap{k}": v for k, v in snaps.items()})
    print(f"    seed {seed}: {done} -> {target} of {steps} steps "
          f"({time.time() - t0:.0f} s), L = {sim.domain_scale():.1f} cells")
    return target, target >= steps


# ======================================================================
def run_coarsening(n, c0, seed, steps, snap_steps, lam_target=12.0,
                   f_scale=100.0, target_amp=1.02, verbose=True):
    """
    Isothermal spinodal decomposition and coarsening on an n x n periodic grid.

    ISOTHERMAL, not cooling. The cooling runs of Section 5 are about arrest; the
    question here is what the coarsening dynamics themselves organise, and a
    collapsing mobility only slows the clock. Fixing M = 1 keeps the morphology
    on a single trajectory and lets snapshots be compared at equal amounts of
    diffusive progress.

    kappa and dt are set exactly as in cooling_2d.run_cooling -- kappa from the
    target spinodal wavelength, dt from the per-step amplification of the
    fastest-growing mode -- so this is the same physical system as the rest of
    the paper, observed with a different instrument.
    """
    free = SphaleriteFreeEnergy(scale=f_scale)
    d2 = free.d2fdc2(c0)
    if d2 >= 0:
        raise ValueError(f"c0 = {c0} is not inside the unstable region")
    kappa = abs(d2) * lam_target ** 2 / (8 * np.pi ** 2)

    kk = np.linspace(1e-3, np.pi, 800)
    k_grow = kk[np.argmax(-kk ** 2 * (d2 + kappa * kk ** 2))]
    den = -k_grow ** 2 * d2 - target_amp * kappa * k_grow ** 4
    dt = (target_amp - 1.0) / den

    sim = CahnHilliard2D(n, 1.0, kappa, free, seed=seed)
    sim.initialise(c0, noise=0.002)
    snaps = {}
    t0 = time.time()
    for s in range(1, steps + 1):
        sim.step(dt, 1.0)
        if s in snap_steps:
            snaps[s] = sim.c.copy()
        if not np.isfinite(sim.c).all():
            raise RuntimeError(f"field diverged at step {s}")
    if verbose:
        print(f"    seed {seed}: {steps:,} steps, {time.time() - t0:.0f} s, "
              f"L = {sim.domain_scale():.1f} cells, kappa = {kappa:.1f}")
    return snaps, kappa, dt


# ======================================================================
# 6. Full diagnostic on one point set
# ======================================================================
def diagnose(pts, L, radii=None, n_modes=48, k_peak=None, label="",
             weights=None, do_nv=True):
    """
    Local index, S(k), and number variance for one configuration.

    do_nv is switched off for the mass-weighted pass: the number variance
    counts points and is blind to the weights, so recomputing it there would
    only duplicate the data curve at several minutes' cost.
    """
    ce = clark_evans(pts, L)
    k, S = structure_factor_points(pts, L, n_modes=n_modes, weights=weights)
    if radii is None:
        radii = np.geomspace(max(ce["d_nn"], 1.0), L / 6.0, 12)
    nv = number_variance(pts, L, radii) if do_nv \
        else np.full(len(radii), np.nan)
    return dict(label=label, ce=ce, k=k, S=S, radii=radii, nv=nv)


def summarise(diag, k_peak, L, d_typ, n_low_shells=5):
    """
    Collapse a diagnostic into the three numbers the verdict rests on:
    the mean of S over the lowest few shells, the low-k exponent, and the
    log-log slope of the number variance.

    The low-k window runs from the smallest available wavevector to 0.3 k_peak
    -- comfortably below the coarsening Bragg peak, where the pattern's own
    periodicity would dominate and the fit would measure the peak rather than
    the k -> 0 limit.
    """
    k, S = diag["k"], diag["S"]
    kk, SS, _ = radial_average(k, S, n_bins=36, k_max=4.0 * k_peak)
    k_min = 2 * np.pi / L
    S_low = float(np.mean(SS[:n_low_shells])) if SS.size else float("nan")
    alpha, alpha_se = low_k_exponent(kk, SS, 0.9 * k_min, 0.45 * k_peak)
    R, nv = np.asarray(diag["radii"]), np.asarray(diag["nv"], float)
    nv = np.where(np.isfinite(nv), nv, 0.0)
    # Fit only where the window is larger than the domain spacing. Below it
    # every point process, hyperuniform or not, has sigma^2 -> <n> ~ R^2 simply
    # because the counts are 0 or 1; the diagnostic has no content there.
    ok = (nv > 0) & (R >= 1.4 * d_typ)
    if ok.sum() < 3:
        ok = nv > 0
    slope = float(np.polyfit(np.log(R[ok]), np.log(nv[ok]), 1)[0]) if ok.sum() > 2 \
        else float("nan")
    S_peak = float(SS.max()) if SS.size else float("nan")
    H = S_low / S_peak if S_peak > 0 else float("nan")
    return dict(N=diag["ce"]["N"], R=diag["ce"]["R"], z=diag["ce"]["z"],
                d_nn=diag["ce"]["d_nn"], S_low=S_low, S_peak=S_peak, H=float(H),
                alpha=alpha, alpha_se=alpha_se, nv_slope=slope,
                classification=classify(S_low, alpha, alpha_se, slope, H),
                k_radial=kk.tolist(), S_radial=SS.tolist())


def classify(S_low, alpha, alpha_se, nv_slope, H):
    """
    Three tiers, not two.

    "hyperuniform" is reserved for a pattern whose S(k) is actually seen to
    FALL towards zero -- a low-k exponent significantly positive and a number
    variance growing more slowly than the area. A pattern can fail that and
    still be far from Poisson: S(k -> 0) may settle on a small plateau, which
    Torquato (2018) calls EFFECTIVELY hyperuniform when the plateau lies below
    about 1e-2 of the peak. The distinction is not pedantry. A plateau means
    the density fluctuations are suppressed by a large constant factor but
    still grow with volume, so the pattern is not in the same class as an
    avian photoreceptor mosaic, and saying so would be wrong.
    """
    if not np.isfinite(alpha):
        return "indeterminate"
    if alpha - 2 * alpha_se > 0.5 and S_low < 0.3 and nv_slope < 1.5:
        return "hyperuniform"
    if np.isfinite(H) and H < 0.05 and S_low < 0.4:
        return "effectively hyperuniform (low-k plateau)"
    if S_low < 0.7:
        return "suppressed but not hyperuniform"
    if S_low > 1.4:
        return "super-Poissonian"
    return "Poisson-like"


# ======================================================================
# 7. Figure
# ======================================================================
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8.5, "pdf.fonttype": 42, "ps.fonttype": 42})
    return plt


def _fmt(x, nd=2):
    """Format a number for a mathtext label without emitting a bare 'nan'
    adjacent to a TeX command, which matplotlib's parser rejects."""
    return "n/a" if not np.isfinite(x) else f"{x:.{nd}f}"


def make_budget_figure(res, outfile, d_typ):
    """
    The variance budget, as the journal's Figure 4.

    Left: what independent domains of the observed size distribution would
    produce, split into its two terms, against what is measured. Right: the
    fraction of that baseline which survives, and the correlation between the
    domain count and the mean domain size in the same window.

    Windows narrower than about 1.4 domain spacings are shaded out rather than
    deleted. There the decomposition is dominated by discreteness -- a window
    holding less than one domain on average has a variance set by whether it
    holds any -- and showing the shaded region is more honest than silently
    starting the axis where the curve becomes well behaved.
    """
    plt = _mpl()
    b = res.get("variance_budget")
    if not b:
        return False
    R = np.array([r["R"] for r in b])
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.4, 3.6))
    lo = 1.4 * d_typ

    ax.loglog(R, [r["baseline"] for r in b], "o-", color="0.25", ms=3.5, lw=1.4,
              label=r"baseline: $\langle N\rangle\,$CV$^2 + $Var$[N]$")
    ax.loglog(R, [r["term_size"] for r in b], "^--", color="#7b3fa0", ms=3,
              lw=1.0, label=r"    size term $\langle N\rangle\,$CV$^2$")
    ax.loglog(R, [r["varN"] for r in b], "v--", color="#c1272d", ms=3, lw=1.0,
              label=r"    count term Var$[N]$")
    ax.loglog(R, [r["varM"] for r in b], "s-", color="#1b7837", ms=3.5, lw=1.6,
              label=r"measured Var$[M]/\langle m\rangle^2$")
    ax.axvspan(R.min(), lo, color="0.9", zorder=0)
    ax.set_xlabel(r"window radius $R$ (cells)")
    ax.set_ylabel("variance in units of one domain")
    ax.legend(fontsize=6.8, frameon=False, loc="upper left")
    ax.set_title("(a) independent domains would fluctuate eightfold more",
                 fontsize=9, loc="left")

    bx.semilogx(R, [r["ratio"] for r in b], "s-", color="#1b7837", ms=3.5,
                lw=1.6, label="measured / baseline")
    bx.semilogx(R, [r["corr_count_size"] for r in b], "^-", color="#8c510a",
                ms=3.5, lw=1.4, label=r"corr$(N, \bar m)$ in the window")
    bx.axhline(1.0, color="0.6", lw=0.7, ls="--")
    bx.axhline(0.0, color="0.6", lw=0.7, ls=":")
    bx.axvspan(R.min(), lo, color="0.9", zorder=0)
    bx.set_ylim(-1.05, 1.15)
    bx.set_xlabel(r"window radius $R$ (cells)")
    bx.set_ylabel("dimensionless")
    bx.legend(fontsize=7, frameon=False, loc="center left")
    bx.set_title("(b) the cancellation deepens with scale", fontsize=9,
                 loc="left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(outfile + "." + ext, bbox_inches="tight", dpi=160)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------
# Journal figures: single- and double-column, grayscale-legible
# ---------------------------------------------------------------------
# PRE column widths. Everything below is sized in inches from these, so the
# figures go into the manuscript at 1:1 and no font is rescaled by LaTeX --
# a figure shrunk by \includegraphics is a figure whose 7 pt labels are no
# longer 7 pt.
COL1, COL2 = 3.386, 6.772          # 8.6 cm and 17.2 cm

# Every series carries a line style AND a marker AND a colour. Colour alone
# fails in monochrome print, and the red/green pairing used in the working
# figures is the worst case: the two have nearly equal luminance, so in
# grayscale they become the same mid-tone line.
STYLE = {
    "data":      dict(color="#c1272d", ls="-",  marker="o", label="centroids (counts)"),
    "data_mass": dict(color="#1b7837", ls="--", marker="s", label="centroids, mass-weighted"),
    "csr":       dict(color="0.35",    ls=":",  marker="x", label="CSR (Poisson)"),
    "rsa":       dict(color="#7b3fa0", ls="-.", marker="^", label="RSA hard core"),
    "lattice":   dict(color="#e08214", ls=(0, (3, 1, 1, 1, 1, 1)), marker="*",
                      label="perturbed lattice"),
}


def _jrc(plt):
    plt.rcParams.update({
        "font.size": 7.5, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.4,
        "lines.linewidth": 1.1, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.linewidth": 0.7,
    })


def _series(ax, key, k, S, logy=True, every=3):
    st = STYLE[key]
    plot = ax.loglog if logy else ax.semilogx
    plot(k, S, ls=st["ls"], color=st["color"], marker=st["marker"],
         markersize=3.0, markevery=every, label=st["label"])


def make_journal_figures(res, outdir, d_typ):
    """
    Split the eight-panel working figure into the four figures of the
    manuscript, at the journal's own column widths and with every series
    distinguished by line style and marker as well as colour.
    """
    plt = _mpl()
    _jrc(plt)
    L, k_peak = res["L"], res["k_peak"]
    made = []

    # ---- Figure 1: morphology and the order plane ----
    fig, (a, e) = plt.subplots(2, 1, figsize=(COL1, COL1 * 1.75))
    fld = np.asarray(res["field_png"])
    p = np.asarray(res["pts_png"])
    win = min(fld.shape[0], 384)
    p = p[(p[:, 0] < win) & (p[:, 1] < win)]
    a.imshow(fld[:win, :win], cmap="gray", interpolation="nearest",
             origin="lower")
    a.plot(p[:, 1], p[:, 0], "o", ms=2.4, mfc="none", mec="#c1272d", mew=0.7)
    a.set_xticks([]); a.set_yticks([])
    a.text(0.02, 0.02, f"{win}$^2$ crop of {int(L)}$^2$", transform=a.transAxes,
           fontsize=6.2, color="w")
    a.set_title("(a)", loc="left")

    e.set_yscale("log")
    e.axhline(1.0, color="0.6", lw=0.6, ls="--")
    e.axvline(1.0, color="0.6", lw=0.6, ls="--")
    for name, (R_, S_) in POCKMARKS.items():
        e.plot(R_, S_, "s", mfc="none", color="0.35", ms=4.5)
    e.annotate("pockmark fields", (0.5, 8.0), fontsize=6.0, color="0.35")
    for key in ("csr", "rsa", "lattice"):
        st, sm = STYLE[key], res["summary"][key]
        e.plot(sm["R"], sm["S_low"], st["marker"], color=st["color"],
               ms=6 if st["marker"] == "*" else 4.5, label=st["label"])
    for key in ("data", "data_mass"):
        st, sm = STYLE[key], res["summary"][key]
        e.plot(sm["R"], sm["S_low"], st["marker"], color=st["color"], ms=6,
               mec="k", mew=0.5, label=st["label"], zorder=5)
    e.set_xlim(0.35, 2.05)
    e.set_xlabel(r"Clark--Evans $R$")
    e.set_ylabel(r"$S(k\to0)$")
    e.legend(frameon=False, loc="lower left", handletextpad=0.4)
    e.set_title("(b)", loc="left")
    fig.tight_layout(pad=0.4)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, "fig_1." + ext), dpi=600 if ext == "png" else None,
                    bbox_inches="tight")
    plt.close(fig); made.append("fig_1")

    # ---- Figure 2: the three spectra ----
    fig, (c, b) = plt.subplots(2, 1, figsize=(COL1, COL1 * 1.75))
    kf = np.asarray(res["k_field"]); Sf = np.asarray(res["S_field"])
    c.loglog(kf, Sf, "-", color="k", lw=1.1, label="concentration field")
    m = (kf > 0.9 * 2 * np.pi / L) & (kf < 0.35 * k_peak)
    if m.sum() > 2:
        A = Sf[m][0] / kf[m][0] ** 4
        c.loglog(kf[m], A * kf[m] ** 4, "--", color="0.5", lw=0.9,
                 label=r"$k^4$")
    c.axvline(k_peak, color="0.7", lw=0.6, ls="-.")
    c.set_xlabel(r"$k$ (cell$^{-1}$)"); c.set_ylabel(r"$S_c(k)$")
    c.legend(frameon=False, loc="lower right")
    c.set_title("(a)", loc="left")

    for key in ("data", "data_mass", "csr", "rsa", "lattice"):
        sm = res["summary"][key]
        _series(b, key, sm["k_radial"], sm["S_radial"])
    b.axhline(1.0, color="0.7", lw=0.6, ls=":")
    b.axvline(k_peak, color="0.7", lw=0.6, ls="-.")
    b.set_xlabel(r"$k$ (cell$^{-1}$)"); b.set_ylabel(r"$S(k)$")
    b.legend(frameon=False, loc="lower right", ncol=1, handletextpad=0.4)
    b.set_title("(b)", loc="left")
    fig.tight_layout(pad=0.4)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, "fig_2." + ext), dpi=600 if ext == "png" else None,
                    bbox_inches="tight")
    plt.close(fig); made.append("fig_2")

    # ---- Figure 3: number variance and the finite-size check ----
    fig, (d, h) = plt.subplots(2, 1, figsize=(COL1, COL1 * 1.75))
    for key in ("data", "csr", "rsa", "lattice"):
        Rw = np.asarray(res["nv_R"][key]); v = np.asarray(res["nv"][key])
        _series(d, key, Rw, v, every=2)
    R0 = np.asarray(res["nv_R"]["data"]); v0 = np.asarray(res["nv"]["data"])
    d.loglog(R0, v0[0] * (R0 / R0[0]) ** 2, "--", color="0.5", lw=0.8)
    d.loglog(R0, v0[0] * (R0 / R0[0]), ":", color="0.5", lw=0.9)
    d.annotate("slope 2", (R0[-1], v0[0] * (R0[-1] / R0[0]) ** 2), fontsize=6,
               color="0.4", ha="right", va="bottom")
    d.annotate("slope 1", (R0[-1], v0[0] * (R0[-1] / R0[0])), fontsize=6,
               color="0.4", ha="right", va="top")
    d.set_xlabel(r"window radius $R$ (cells)")
    d.set_ylabel(r"$\sigma^2(R)$")
    d.legend(frameon=False, loc="upper left", handletextpad=0.4)
    d.set_title("(a)", loc="left")

    hd = res["summary"]["data"]
    h.loglog(hd["k_radial"], hd["S_radial"], "-", color="#c1272d", marker="o",
             ms=3, markevery=3, label=f"${int(res['n'])}^2$ box")
    cmp_ = res.get("compare")
    if cmp_:
        h.loglog(cmp_["k_radial"], cmp_["S_radial"], "--", color="0.35",
                 marker="s", ms=3, markevery=3, label=f"${cmp_['grid']}^2$ box")
        h.axvline(cmp_["k_min"], color="0.35", lw=0.6, ls=":")
    h.axvline(2 * np.pi / L, color="#c1272d", lw=0.6, ls=":")
    h.axhline(1.0, color="0.7", lw=0.6, ls=":")
    h.set_xlabel(r"$k$ (cell$^{-1}$)"); h.set_ylabel(r"$S(k)$, counts")
    h.legend(frameon=False, loc="upper left")
    h.set_title("(b)", loc="left")
    fig.tight_layout(pad=0.4)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, "fig_3." + ext), dpi=600 if ext == "png" else None,
                    bbox_inches="tight")
    plt.close(fig); made.append("fig_3")

    # ---- Figure 4: the variance budget, double column ----
    bud = res.get("variance_budget")
    if bud:
        Rw = np.array([r["R"] for r in bud])
        lo = 1.4 * d_typ
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(COL2, COL2 * 0.36))
        ax.loglog(Rw, [r["baseline"] for r in bud], "-", color="k", marker="o",
                  ms=3, label=r"baseline $\langle N\rangle{\rm CV}^2+{\rm Var}[N]$")
        ax.loglog(Rw, [r["term_size"] for r in bud], "--", color="#7b3fa0",
                  marker="^", ms=2.6, lw=0.9,
                  label=r"$\quad$size term $\langle N\rangle{\rm CV}^2$")
        ax.loglog(Rw, [r["varN"] for r in bud], "-.", color="#c1272d",
                  marker="v", ms=2.6, lw=0.9, label=r"$\quad$count term ${\rm Var}[N]$")
        ax.loglog(Rw, [r["varM"] for r in bud], ":", color="#1b7837",
                  marker="s", ms=3, lw=1.4,
                  label=r"measured ${\rm Var}[M]/\langle m\rangle^2$")
        ax.axvspan(Rw.min(), lo, color="0.88", zorder=0)
        ax.set_xlabel(r"window radius $R$ (cells)")
        ax.set_ylabel("variance in units of one domain")
        ax.legend(frameon=False, loc="upper left", handletextpad=0.4)
        ax.set_title("(a)", loc="left")

        bx.semilogx(Rw, [r["ratio"] for r in bud], ":", color="#1b7837",
                    marker="s", ms=3, lw=1.4, label="measured / baseline")
        bx.semilogx(Rw, [r["corr_count_size"] for r in bud], "-",
                    color="#8c510a", marker="D", ms=3, lw=1.1,
                    label=r"${\rm corr}(N,\bar m)$")
        bx.axhline(1.0, color="0.6", lw=0.6, ls="--")
        bx.axhline(0.0, color="0.6", lw=0.6, ls=":")
        bx.axvspan(Rw.min(), lo, color="0.88", zorder=0)
        bx.set_ylim(-1.05, 1.15)
        bx.set_xlabel(r"window radius $R$ (cells)")
        bx.set_ylabel("dimensionless")
        bx.legend(frameon=False, loc="center left", handletextpad=0.4)
        bx.set_title("(b)", loc="left")
        fig.tight_layout(pad=0.4)
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(outdir, "fig_4." + ext),
                        dpi=600 if ext == "png" else None, bbox_inches="tight")
        plt.close(fig); made.append("fig_4")
    return made


def _finish(fig, outfile, plt):
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    for ext in ("pdf", "png"):
        fig.savefig(outfile + "." + ext, bbox_inches="tight", dpi=160)
    plt.close(fig)


def make_figure(res, outfile):
    plt = _mpl()
    L = res["L"]
    k_peak = res["k_peak"]
    fig = plt.figure(figsize=(14.6, 7.2))
    gs = fig.add_gridspec(2, 4, hspace=0.36, wspace=0.30)

    # (a) morphology with centroids
    a = fig.add_subplot(gs[0, 0])
    # A 1024^2 field drawn 4 cm wide is a texture, not a picture: at that
    # reduction the domains are two pixels across and the centroid markers
    # cover them. A fixed-size crop is shown instead, so that the panel means
    # the same thing whatever the box size, and the crop is stated.
    fld = np.asarray(res["field_png"])
    p = np.asarray(res["pts_png"])
    win = min(fld.shape[0], 384)
    fld = fld[:win, :win]
    p = p[(p[:, 0] < win) & (p[:, 1] < win)]
    a.imshow(fld, cmap="magma", interpolation="nearest", origin="lower")
    a.plot(p[:, 1], p[:, 0], "o", ms=2.6, mfc="none", mec="#39d0ff", mew=0.8)
    a.set_xticks([]); a.set_yticks([])
    a.set_title(f"(a) In-rich domains and centroids\n"
                f"{int(res['N_mean'])} domains, spacing "
                f"{res['d_nn']:.0f} cells", fontsize=8.5, loc="left")
    a.text(0.02, 0.02, f"{win}$^2$ crop", transform=a.transAxes, fontsize=6.5,
           color="w")

    # (b) S(k) of the point pattern
    b = fig.add_subplot(gs[0, 1])
    for key, col, ls, lab in (("data", "#c1272d", "-", "centroids (counts)"),
                              ("data_mass", "#1b7837", "-", "centroids, mass-weighted"),
                              ("csr", "0.35", "-", "CSR (Poisson)"),
                              ("rsa", "#7b3fa0", "-", "RSA hard-core"),
                              ("lattice", "#e08214", "--", "perturbed lattice")):
        sm = res["summary"][key]
        b.loglog(sm["k_radial"], sm["S_radial"], ls, color=col, lw=1.3, label=lab)
    b.axhline(1.0, color="0.6", lw=0.7, ls=":")
    b.axvline(k_peak, color="0.6", lw=0.7, ls="-.")
    b.set_xlabel(r"$k$ (cell$^{-1}$)")
    b.set_ylabel(r"$S(k)$")
    b.set_title("(b) point pattern, counts vs solute mass\n"
                + r"$\alpha_{\rm count} = $"
                + _fmt(res["summary"]["data"]["alpha"])
                + r",  $\alpha_{\rm mass} = $"
                + _fmt(res["summary"]["data_mass"]["alpha"])
                + "   (Poisson: 0)", fontsize=8.5, loc="left")
    b.legend(fontsize=6.2, frameon=False, loc="upper left")

    # (c) S(k) of the field
    c = fig.add_subplot(gs[0, 2])
    c.loglog(res["k_field"], res["S_field"], "-", color="#1b7837", lw=1.3,
             label="concentration field")
    kf = np.asarray(res["k_field"])
    m = (kf > 0.9 * 2 * np.pi / L) & (kf < 0.35 * k_peak)
    if m.sum() > 2:
        A = np.asarray(res["S_field"])[m][0] / kf[m][0] ** 4
        c.loglog(kf[m], A * kf[m] ** 4, "k--", lw=0.8, label=r"$k^4$ guide")
    c.axvline(k_peak, color="0.6", lw=0.7, ls="-.")
    c.set_xlabel(r"$k$ (cell$^{-1}$)")
    c.set_ylabel(r"$S_c(k)$")
    c.set_title("(c) concentration field: " + r"$\alpha_c = $"
                + _fmt(res["alpha_field"])
                + "\nconserved dynamics suppress low $k$",
                fontsize=8.5, loc="left")
    c.legend(fontsize=6.5, frameon=False, loc="lower right")

    # (d) number variance
    d = fig.add_subplot(gs[1, 0])
    for key, col, lab in (("data", "#c1272d", "domain centroids"),
                          ("csr", "0.35", "CSR"),
                          ("rsa", "#7b3fa0", "RSA hard-core"),
                          ("lattice", "#e08214", "perturbed lattice")):
        R = np.asarray(res["nv_R"][key]); v = np.asarray(res["nv"][key])
        d.loglog(R, v, "o-", color=col, ms=2.5, lw=1.2, label=lab)
    R0 = np.asarray(res["nv_R"]["data"]); v0 = np.asarray(res["nv"]["data"])
    d.loglog(R0, v0[0] * (R0 / R0[0]) ** 2, "k--", lw=0.8, label="slope 2 (area)")
    d.loglog(R0, v0[0] * (R0 / R0[0]) ** 1, "k:", lw=0.9, label="slope 1 (perimeter)")
    from matplotlib.ticker import FuncFormatter, NullFormatter
    d.xaxis.set_minor_formatter(NullFormatter())
    d.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    _r = np.asarray(res["nv_R"]["data"])
    d.set_xticks([t for t in (5, 10, 20, 40, 80, 160)
                  if _r.min() <= t <= _r.max()])
    d.set_xlabel(r"window radius $R$ (cells)")
    d.set_ylabel(r"$\sigma^2(R)$")
    d.set_title("(d) number variance, slope "
                + _fmt(res["summary"]["data"]["nv_slope"])
                + "\nfitted above the domain spacing", fontsize=8.5, loc="left")
    d.legend(fontsize=6.2, frameon=False, loc="upper left")

    # (e) order plane, with the pockmark fields
    e = fig.add_subplot(gs[1, 1])
    e.axhline(1.0, color="0.6", lw=0.7, ls="--")
    e.axvline(1.0, color="0.6", lw=0.7, ls="--")
    e.set_yscale("log")
    ylo = 10 ** np.floor(np.log10(
        max(min(res["summary"]["data"]["S_low"],
                res["summary"]["data_mass"]["S_low"]), 1e-4))) / 3
    e.axhspan(ylo, 1.0, color="#fff2df", zorder=0)
    for name, (R, S) in POCKMARKS.items():
        e.plot(R, S, "s", color="#4d4d4d", ms=5)
        e.annotate(name, (R, S), fontsize=6.0, color="#4d4d4d",
                   xytext=(5, -3), textcoords="offset points")
    for key, col, mk, lab, off in (
            ("csr", "0.35", "s", "CSR", (6, -3)),
            ("rsa", "#7b3fa0", "^", "RSA hard-core", (-52, 4)),
            ("lattice", "#e08214", "*", "perturbed lattice", (-64, -3))):
        sk = res["summary"][key]
        e.plot(sk["R"], sk["S_low"], mk, color=col, ms=8 if mk == "*" else 5)
        e.annotate(lab, (sk["R"], sk["S_low"]), fontsize=6.0, color=col,
                   xytext=off, textcoords="offset points")
    sd = res["summary"]["data"]
    e.plot(sd["R"], sd["S_low"], "o", color="#c1272d", ms=8, mec="k", mew=0.6,
           zorder=5)
    e.annotate("In domains\n(counts)", (sd["R"], sd["S_low"]), fontsize=6.8,
               color="#c1272d", xytext=(-26, 12), textcoords="offset points")
    e.set_xlim(0.35, 2.05)
    e.set_xlabel("Clark-Evans $R$  (<1 clustered, >1 regular)")
    e.set_ylabel(r"$S(k\to 0)$")
    smw = res["summary"]["data_mass"]
    e.plot(smw["R"], smw["S_low"], "D", color="#1b7837", ms=6, mec="k", mew=0.5,
           zorder=5)
    e.annotate("same points,\nmass-weighted", (smw["R"], smw["S_low"]),
               fontsize=6.5, color="#1b7837", xytext=(6, -12),
               textcoords="offset points")
    e.set_title("(e) order plane: pockmark fields (grey)\n"
                "against the phase-field domains", fontsize=8.5, loc="left")

    # (g) polydispersity: monodisperse subset vs matched random thinning
    g = fig.add_subplot(gs[0, 3])
    cv = res.get("count_vs_mass")
    if cv:
        Rw = np.asarray(cv["radii"])
        g.loglog(Rw, cv["var_count"], "o-", color="#c1272d", ms=3, lw=1.3,
                 label=r"domain count, $\sigma^2_N$")
        g.loglog(Rw, cv["var_mass"], "s-", color="#1b7837", ms=3, lw=1.3,
                 label=r"solute mass, $\sigma^2_M/\langle m\rangle^2$")
        g.loglog(Rw, cv["var_count"][0] * (Rw / Rw[0]) ** 2, "k--", lw=0.8,
                 label="slope 2 (area)")
        g.loglog(Rw, cv["var_count"][0] * (Rw / Rw[0]), "k:", lw=0.9,
                 label="slope 1 (perimeter)")
        g.set_xlabel(r"window radius $R$ (cells)")
        g.set_ylabel("variance in units of one domain")
        g.legend(fontsize=6.0, frameon=False, loc="upper left")
        # The correlation between the two quantities in the same windows is
        # where the compensation becomes visible: positive at small R, where a
        # window holding one more domain simply holds one more domain's worth
        # of solute, and NEGATIVE at large R, where a region with a surplus of
        # domains turns out to hold slightly less solute per domain. A count
        # excess paid for by a size deficit is exactly what a conserved field
        # with a free domain count is expected to produce.
        g2 = g.twinx()
        g2.semilogx(Rw, cv["corr"], "^-", color="#8c510a", ms=3, lw=1.0)
        g2.axhline(0.0, color="#8c510a", lw=0.6, ls=":")
        g2.set_ylim(-0.8, 1.05)
        g2.set_ylabel(r"corr$(N, M)$ in the same window", color="#8c510a",
                      fontsize=7.5)
        g2.tick_params(axis="y", labelcolor="#8c510a", labelsize=7)
        g2.text(Rw[-1], cv["corr"][-1] - 0.16, "count excess paid for\nby a size "
                "deficit", fontsize=6.0, color="#8c510a", ha="right")
    g.set_title("(g) same windows, two quantities: the\nsolute fluctuates less "
                "than the count", fontsize=8.5, loc="left")

    # (h) finite size: the same measurement in a box twice as wide
    h = fig.add_subplot(gs[1, 2])
    cmp_ = res.get("compare")
    hd = res["summary"]["data"]
    h.loglog(hd["k_radial"], hd["S_radial"], "-", color="#c1272d", lw=1.4,
             label=f"{int(res['n'])}$^2$ box" if "n" in res else "this box")
    if cmp_:
        h.loglog(cmp_["k_radial"], cmp_["S_radial"], "-", color="#2166ac",
                 lw=1.4, label=cmp_["label"])
        h.axvline(cmp_["k_min"], color="#2166ac", lw=0.7, ls=":")
    h.axvline(2 * np.pi / L, color="#c1272d", lw=0.7, ls=":")
    h.axhline(1.0, color="0.6", lw=0.7, ls=":")
    h.set_xlabel(r"$k$ (cell$^{-1}$)")
    h.set_ylabel(r"$S(k)$, domain counts")
    h.set_title("(h) finite-size check: does the low-$k$\nupturn survive a "
                "wider box?", fontsize=8.5, loc="left")
    h.legend(fontsize=6.5, frameon=False, loc="upper left")

    # (f) threshold robustness
    f = fig.add_subplot(gs[1, 3])
    th = res["threshold_scan"]
    th_from = ""
    if not th["quantile"] and res.get("compare", {}) and \
            res["compare"].get("threshold_scan", {}).get("quantile"):
        th = res["compare"]["threshold_scan"]
        th_from = f" ({res['compare']['grid']}$^2$ run)"
    if not th["quantile"]:
        f.text(0.5, 0.5, "threshold scan not run\n(see the 512$^2$ result)",
               ha="center", va="center", fontsize=7.5, color="0.45",
               transform=f.transAxes)
        f.set_xticks([]); f.set_yticks([])
        f.set_title("(f) threshold sensitivity", fontsize=8.5, loc="left")
        _finish(fig, outfile, plt)
        return
    ax2 = f.twinx()
    f.plot(th["quantile"], th["alpha"], "o-", color="#c1272d", ms=3, lw=1.2,
           label=r"$\alpha$")
    f.plot(th["quantile"], th["nv_slope"], "s-", color="#1b7837", ms=3, lw=1.2,
           label="NV slope")
    ax2.plot(th["quantile"], th["N"], "^--", color="0.5", ms=3, lw=1.0,
             label="N points")
    f.axhline(0.0, color="0.7", lw=0.7, ls="-")
    f.axhline(1.0, color="0.7", lw=0.7, ls=":")
    f.set_xlabel("threshold position between the two phases")
    f.set_ylabel(r"$\alpha$ / number-variance slope")
    ax2.set_ylabel("domains detected", color="0.5")
    # zero-based, so that a 1 % drift in the count is seen as a 1 % drift
    ax2.set_ylim(0, 1.35 * max(th["N"]))
    f.set_ylim(-1.0, 2.6)
    f.set_title("(f) the verdict does not depend\non where the phases are cut"
                + th_from, fontsize=8.5, loc="left")
    f.text(0.03, 0.06, r"$\alpha=0$: Poisson;  slope 2: area-like",
           transform=f.transAxes, fontsize=6.0, color="0.4")
    h1, l1 = f.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    f.legend(h1 + h2, l1 + l2, fontsize=6.2, frameon=False, loc="lower right")

    fig.suptitle("Spatial order of the phase-field indium nanodomains: strong "
                 "local regularity, strongly suppressed long-range "
                 "fluctuations,\nand a conserved quantity that is the solute "
                 "rather than the domain count. CSR in the same box validates "
                 "the estimators.", fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    for ext in ("pdf", "png"):
        fig.savefig(outfile + "." + ext, bbox_inches="tight", dpi=160)
    plt.close(fig)


# ======================================================================
# 8. Driver
# ======================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="grid size")
    ap.add_argument("--seeds", type=int, default=None, help="independent runs")
    ap.add_argument("--steps", type=int, default=None, help="coarsening steps")
    ap.add_argument("--c0", type=float, default=0.30,
                    help="mean composition (must be inside the spinodal)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--cache", default=None,
                    help="npz file of snapshot fields; written if absent, "
                         "read if present. Lets a long run be simulated once "
                         "and re-analysed many times.")
    ap.add_argument("--seed0", type=int, default=0,
                    help="first RNG seed; lets a long ensemble be simulated in "
                         "batches that are merged afterwards")
    ap.add_argument("--simulate-only", action="store_true",
                    help="fill the cache and stop")
    ap.add_argument("--snaps", default=None,
                    help="comma-separated step numbers to snapshot "
                         "(default: steps/4, steps/2, steps)")
    ap.add_argument("--stages", default=None,
                    help="comma-separated snapshot steps to ANALYSE "
                         "(default: all of them). Analysis of a 1024^2 "
                         "ensemble costs minutes per stage.")
    ap.add_argument("--no-scan", action="store_true",
                    help="skip the threshold-sensitivity scan")
    ap.add_argument("--chunk", type=int, default=0,
                    help="with --simulate-only: advance each run by at most "
                         "this many steps, checkpoint, and exit. Repeat the "
                         "same command until it reports every run finished.")
    ap.add_argument("--slow-fft", action="store_true",
                    help="use numpy's FFT instead of scipy's (identical "
                         "arithmetic, about 1.7x slower at 1024^2)")
    ap.add_argument("--compare", default=None,
                    help="a hyperuniformity.json from a SMALLER box, plotted "
                         "alongside for the finite-size check")
    ap.add_argument("--figure-only", action="store_true",
                    help="redraw the figure from data/hyperuniformity.json and "
                         "the field cache, without recomputing anything")
    args = ap.parse_args()

    n = args.n or (256 if args.quick else 512)
    n_seeds = args.seeds or (2 if args.quick else 4)
    steps = args.steps or (2000 if args.quick else 4000)
    snap_steps = sorted({int(x) for x in args.snaps.split(",")}) if args.snaps \
        else sorted({steps // 4, steps // 2, steps})
    stages = sorted({int(x) for x in args.stages.split(",")}) if args.stages \
        else list(snap_steps)
    L = float(n)

    os.makedirs(_OUT, exist_ok=True)
    os.makedirs(_DATA, exist_ok=True)

    print("Hyperuniformity of the phase-field indium nanodomains")
    print(f"  grid {n} x {n}, c0 = {args.c0}, {n_seeds} seeds, "
          f"{steps:,} steps, snapshots at {snap_steps}\n")

    if args.figure_only:
        _redraw.compare = args.compare
        _redraw(cache=args.cache, steps=steps, snap_steps=stages, L=L)
        return

    # ---- 1. simulate (or reload) ----
    fields = {s: [] for s in snap_steps}
    cache = args.cache
    if cache and os.path.exists(cache):
        z = np.load(cache)
        for s in snap_steps:
            fields[s] = [z[f"s{s}_{i}"] for i in range(n_seeds)]
        print(f"  fields reloaded from {cache}\n")
    elif args.chunk:
        # chunked, checkpointed simulation: run this command repeatedly
        fast = not args.slow_fft
        if fast:
            free, kappa, dt, _ = _solver_setup(n, args.c0, 0, fast=False)
            d = verify_fft_backend(free, kappa, dt)
            print(f"  fast FFT backend verified against numpy "
                  f"(max divergence {d:.1e} after 25 steps)")
        base = os.path.splitext(cache)[0]
        finished = []
        for seed in range(args.seed0, args.seed0 + n_seeds):
            _, ok = simulate_chunked(n, args.c0, seed, steps, snap_steps,
                                     f"{base}.seed{seed}.ckpt.npz", args.chunk,
                                     fast=fast)
            finished.append(ok)
        if all(finished):
            out = {}
            for i, seed in enumerate(range(args.seed0, args.seed0 + n_seeds)):
                z = np.load(f"{base}.seed{seed}.ckpt.npz")
                for st in snap_steps:
                    out[f"s{st}_{i}"] = z[f"snap{st}"]
            np.savez_compressed(cache, **out)
            print(f"\n  all {n_seeds} runs complete; fields written to {cache}")
        else:
            print(f"\n  not finished -- run the same command again")
        return
    else:
        for seed in range(args.seed0, args.seed0 + n_seeds):
            snaps, kappa, dt = run_coarsening(n, args.c0, seed, steps, snap_steps)
            for s in snap_steps:
                fields[s].append(snaps[s])
        if cache:
            np.savez_compressed(cache, **{f"s{s}_{i}": fields[s][i]
                                          for s in snap_steps
                                          for i in range(n_seeds)})
            print(f"  fields cached to {cache}")
    if args.simulate_only:
        return

    stage_out = {}
    final = stages[-1]

    for stage in stages:
        # ---- 2. point sets, pooled over seeds ----
        sets, masses, thr_used = [], [], []
        for c in fields[stage]:
            thr = phase_threshold(c)
            pts, areas, mass = domain_centroids(c, thr)
            sets.append(pts)
            masses.append(mass)
            thr_used.append(thr)
        Ns = [len(p) for p in sets]
        N_mean = float(np.mean(Ns))
        rho = N_mean / L ** 2
        d_typ = 1.0 / np.sqrt(rho)
        k_peak = 2 * np.pi / d_typ

        rng = np.random.default_rng(12345)
        # exclusion for the RSA null: the closest approach actually observed
        d_ex = float(np.mean([clark_evans(p, L)["d_nn_min"] for p in sets]))

        radii = np.geomspace(0.5 * d_typ, L / 6.0, 12)
        pools = {k: dict(k=[], S=[], nv=[], R=[], z=[], N=[])
                 for k in ("data", "data_mass", "csr", "rsa", "lattice")}

        for pts, mass in zip(sets, masses):
            N = len(pts)
            models = {
                "data": (pts, None, True),
                "data_mass": (pts, mass, False),
                "csr": (null_csr(N, L, rng), None, True),
                "rsa": (null_rsa(N, L, d_ex, rng)[0], None, True),
                "lattice": (null_perturbed_lattice(N, L, 0.20, rng), None, True),
            }
            n_modes = int(min(128, max(48, 4.0 * np.sqrt(max(N, 1)))))
            for key, (P, W, do_nv) in models.items():
                dg = diagnose(P, L, radii=radii, label=key, weights=W,
                              n_modes=n_modes, do_nv=do_nv)
                pools[key]["k"].append(dg["k"])
                pools[key]["S"].append(dg["S"])
                pools[key]["nv"].append(dg["nv"])
                pools[key]["R"].append(dg["ce"]["R"])
                pools[key]["z"].append(dg["ce"]["z"])
                pools[key]["N"].append(dg["ce"]["N"])

        summary = {}
        nv_out, nvR_out = {}, {}
        for key, P in pools.items():
            diag = dict(k=np.concatenate(P["k"]), S=np.concatenate(P["S"]),
                        radii=radii, nv=np.mean(P["nv"], axis=0),
                        ce=dict(N=int(np.mean(P["N"])),
                                R=float(np.mean(P["R"])),
                                z=float(np.mean(P["z"])),
                                d_nn=float(d_typ)))
            summary[key] = summarise(diag, k_peak, L, d_typ)
            nv_out[key] = diag["nv"].tolist()
            nvR_out[key] = radii.tolist()

        # ---- 3. the field itself ----
        kf_all, Sf_all = [], []
        for c in fields[stage]:
            kf, Sf = structure_factor_field(c)
            kf_all.append(kf); Sf_all.append(Sf)
        kf, Sf, _ = radial_average(np.concatenate(kf_all),
                                   np.concatenate(Sf_all),
                                   n_bins=60, k_max=4.0 * k_peak)
        alpha_field, alpha_field_se = low_k_exponent(kf, Sf, 0.9 * 2 * np.pi / L,
                                                     0.35 * k_peak)

        cv_n, cv_m, cv_corr = [], [], []
        for pts, mass in zip(sets, masses):
            a, b, c_ = count_vs_mass_variance(pts, mass, L, radii)
            cv_n.append(a); cv_m.append(b); cv_corr.append(c_)
        cvar = dict(radii=radii.tolist(),
                    var_count=np.mean(cv_n, axis=0).tolist(),
                    var_mass=np.mean(cv_m, axis=0).tolist(),
                    corr=np.mean(cv_corr, axis=0).tolist())
        _ok = np.asarray(cvar["radii"]) >= 1.4 * d_typ
        cvar["slope_count"] = float(np.polyfit(
            np.log(np.asarray(cvar["radii"])[_ok]),
            np.log(np.asarray(cvar["var_count"])[_ok]), 1)[0])
        cvar["slope_mass"] = float(np.polyfit(
            np.log(np.asarray(cvar["radii"])[_ok]),
            np.log(np.asarray(cvar["var_mass"])[_ok]), 1)[0])
        cvar["ratio_at_max_R"] = float(cvar["var_mass"][-1] / cvar["var_count"][-1])

        budgets = [variance_budget(pts, mass, L, radii)
                   for pts, mass in zip(sets, masses)]
        budget = []
        for i in range(len(radii)):
            keys = budgets[0][i].keys()
            budget.append({k: float(np.mean([b[i][k] for b in budgets]))
                           for k in keys})

        poly = polydispersity_test(sets, masses, L,
                                   n_modes=int(min(128, max(48, 4.0 * np.sqrt(
                                       max(N_mean, 1))))))

        stage_out[stage] = dict(
            polydispersity=poly, count_vs_mass=cvar,
            variance_budget=budget,
            N_mean=N_mean, k_peak=float(k_peak), d_typ=float(d_typ),
            d_exclusion=d_ex, threshold=float(np.mean(thr_used)),
            summary=summary, nv=nv_out, nv_R=nvR_out,
            k_field=kf.tolist(), S_field=Sf.tolist(),
            alpha_field=float(alpha_field), alpha_field_se=float(alpha_field_se))

        s, sm = summary["data"], summary["data_mass"]
        print(f"  stage {stage:>6,} steps: N = {N_mean:6.1f}   "
              f"R = {s['R']:.2f} (z = {s['z']:+.1f})")
        print(f"      counts:        S(k->0) = {s['S_low']:.3f}  "
              f"alpha = {s['alpha']:+.2f}+/-{s['alpha_se']:.2f}  "
              f"NV slope = {s['nv_slope']:.2f}  H = {s['H']:.3f}  "
              f"-> {s['classification']}")
        print(f"      solute mass:   S(k->0) = {sm['S_low']:.3f}  "
              f"alpha = {sm['alpha']:+.2f}+/-{sm['alpha_se']:.2f}  "
              f"H = {sm['H']:.3f}  -> {sm['classification']}")
        print(f"      field:         alpha_c = {alpha_field:+.2f}"
              f"+/-{alpha_field_se:.2f}")
        print("      variance budget (units of one domain):")
        print("        R      <N>    Var[N]  <N>CV^2   baseline  Var[M]  "
              "residual  cov(N,size)")
        for b_ in budget:
            print(f"        {b_['R']:6.1f} {b_['meanN']:7.1f} {b_['varN']:8.2f} "
                  f"{b_['term_size']:8.2f} {b_['baseline']:10.2f} "
                  f"{b_['varM']:7.2f} {b_['residual']:9.2f} "
                  f"{b_['cov_count_size']:+11.3f}")
        print(f"      window variance: count slope {cvar['slope_count']:.2f}, "
              f"solute-mass slope {cvar['slope_mass']:.2f}, "
              f"mass/count variance at the widest window "
              f"{cvar['ratio_at_max_R']:.2f}")
        for r in poly:
            print(f"      monodisperse subset q = {r['retention']:.2f} "
                  f"(mass CV {r['cv_all']:.2f} -> {r['cv_kept']:.2f}): "
                  f"S(k->0) = {r['filtered_S_low']:.3f} "
                  f"(random thinning to the same N: "
                  f"{r['thinned_S_low']:.3f});  "
                  f"Clark-Evans {r['filtered_R']:.2f} vs {r['thinned_R']:.2f}")

    # ---- 4. threshold robustness, at the final stage ----
    scan = dict(quantile=[], alpha=[], nv_slope=[], N=[], R=[])
    if not args.no_scan:
        print("\n  threshold scan (final stage)")
    for q in (() if args.no_scan else (0.30, 0.40, 0.50, 0.60, 0.70)):
        ks, Ss, nvs, Rs, Ns = [], [], [], [], []
        for c in fields[final]:
            lo, hi = np.percentile(c, 2), np.percentile(c, 98)
            thr = lo + q * (hi - lo)
            pts, _, _ = domain_centroids(c, thr)
            if len(pts) < 20:
                continue
            dg = diagnose(pts, L, radii=np.geomspace(
                0.5 * L / np.sqrt(len(pts)), L / 6.0, 12))
            ks.append(dg["k"]); Ss.append(dg["S"]); nvs.append(dg["nv"])
            Rs.append(dg["ce"]["R"]); Ns.append(len(pts))
        if not ks:
            continue
        kp = 2 * np.pi * np.sqrt(np.mean(Ns)) / L
        diag = dict(k=np.concatenate(ks), S=np.concatenate(Ss),
                    radii=np.geomspace(0.5 * L / np.sqrt(np.mean(Ns)), L / 6.0, 12),
                    nv=np.mean(nvs, axis=0),
                    ce=dict(N=int(np.mean(Ns)), R=float(np.mean(Rs)), z=0.0,
                            d_nn=float(L / np.sqrt(np.mean(Ns)))))
        sm = summarise(diag, kp, L, float(L / np.sqrt(np.mean(Ns))))
        scan["quantile"].append(q); scan["alpha"].append(sm["alpha"])
        scan["nv_slope"].append(sm["nv_slope"]); scan["N"].append(float(np.mean(Ns)))
        scan["R"].append(sm["R"])
        print(f"    cut at {q:.2f} of the composition gap: N = {np.mean(Ns):6.1f}, "
              f"R = {sm['R']:.2f}, alpha = {sm['alpha']:+.2f}, "
              f"NV slope = {sm['nv_slope']:.2f}")

    # ---- 5. figure and json ----
    c_show = fields[final][0]
    thr = phase_threshold(c_show)
    pts_show, _, _ = domain_centroids(c_show, thr)
    res = dict(stage_out[final])
    res.update(L=L, n=n, c0=args.c0, seeds=n_seeds, steps=steps,
               field_png=c_show, pts_png=pts_show,
               d_nn=stage_out[final]["d_typ"], threshold_scan=scan,
               compare=_load_compare(args.compare, final))
    make_figure(res, os.path.join(_OUT, "fig_hyperuniformity"))
    made = make_journal_figures(res, _OUT, stage_out[final]["d_typ"])
    print(f"  journal figures -> {', '.join(made)}")
    if make_budget_figure(res, os.path.join(_OUT, "fig_variance_budget"),
                          stage_out[final]["d_typ"]):
        print(f"  figures -> {os.path.join(_OUT, 'fig_variance_budget.png')}")

    out = dict(
        grid=n, c0=args.c0, seeds=n_seeds, steps=steps, box_L=L,
        stages={str(s): {k: v for k, v in stage_out[s].items()}
                for s in stages},
        threshold_scan=scan,
        verdict=_verdict(stage_out[final]["summary"]))
    with open(os.path.join(_DATA, "hyperuniformity.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    print("\n" + _report(stage_out[final], L))
    print(f"\n  figures -> {os.path.join(_OUT, 'fig_hyperuniformity.png')}")
    print(f"  data    -> {os.path.join(_DATA, 'hyperuniformity.json')}")


def _load_compare(path, stage):
    """
    Pull the count spectrum of a smaller-box run for the finite-size panel.

    The comparison is only meaningful at the SAME coarsening stage: the
    spectrum depends on how far the pattern has coarsened, so a 1024^2 run at
    2000 steps must be set against a 512^2 run at 2000 steps and not against
    whatever stage that file happens to end on.
    """
    if not path or not os.path.exists(path):
        return None
    with open(path) as fh:
        other = json.load(fh)
    key = str(stage)
    if key not in other["stages"]:
        print(f"  note: {path} has no stage {stage}; finite-size panel skipped")
        return None
    sm = other["stages"][key]["summary"]["data"]
    n_other = other.get("grid", "?")
    return dict(threshold_scan=other.get("threshold_scan", {}),
                k_radial=sm["k_radial"], S_radial=sm["S_radial"],
                S_low=sm["S_low"], alpha=sm["alpha"], alpha_se=sm["alpha_se"],
                label=f"{n_other}$^2$ box", grid=n_other,
                k_min=2 * np.pi / float(other["box_L"]))


def _redraw(cache, steps, snap_steps, L):
    """Rebuild the figure from the saved results, for cosmetic iteration."""
    with open(os.path.join(_DATA, "hyperuniformity.json")) as fh:
        out = json.load(fh)
    final = str(snap_steps[-1])
    res = dict(out["stages"][final])
    c_show = np.load(cache)[f"s{final}_0"]
    thr = phase_threshold(c_show)
    pts_show, _, _ = domain_centroids(c_show, thr)
    res.update(L=L, n=out.get("grid"), field_png=c_show, pts_png=pts_show,
               d_nn=res["d_typ"], threshold_scan=out["threshold_scan"],
               compare=_load_compare(_redraw.compare, int(final)))
    make_figure(res, os.path.join(_OUT, "fig_hyperuniformity"))
    make_budget_figure(res, os.path.join(_OUT, "fig_variance_budget"),
                       res["d_typ"])
    print("  journal figures -> "
          + ", ".join(make_journal_figures(res, _OUT, res["d_typ"])))
    print("  figure redrawn from data/hyperuniformity.json")


def _verdict(summary):
    csr = summary["csr"]
    control_ok = (abs(csr["R"] - 1) < 0.08 and abs(csr["S_low"] - 1) < 0.20
                  and abs(csr["nv_slope"] - 2) < 0.30)
    return dict(csr_control_passes=bool(control_ok),
                domain_counts=summary["data"]["classification"],
                solute_mass=summary["data_mass"]["classification"])


def _report(stage, L):
    su = stage["summary"]
    s, sm, csr, rsa, lat = (su["data"], su["data_mass"], su["csr"], su["rsa"],
                            su["lattice"])
    v = _verdict(su)
    lines = [
        "  " + "-" * 72,
        "  CONTROL   CSR through the identical pipeline: "
        f"R = {csr['R']:.2f}, S(k->0) = {csr['S_low']:.2f}, "
        f"NV slope = {csr['nv_slope']:.2f}",
        f"            (expected 1.00, 1.00, 2.00)  ->  "
        f"{'estimators validated' if v['csr_control_passes'] else 'CHECK THE PIPELINE'}",
        f"            perturbed lattice, a known hyperuniform pattern, returns "
        f"alpha = {lat['alpha']:+.2f}, NV slope = {lat['nv_slope']:.2f}",
        "",
        "  LOCAL     Clark-Evans "
        f"R = {s['R']:.2f} (z = {s['z']:+.1f}). The hard-core RSA null reaches "
        f"only {rsa['R']:.2f},",
        "            so the domain spacing is more regular than exclusion alone "
        "can make it.",
        "",
        "  LONG      domain COUNTS:  "
        f"S(k->0) = {s['S_low']:.3f}, alpha = {s['alpha']:+.2f} "
        f"+/- {s['alpha_se']:.2f}, NV slope = {s['nv_slope']:.2f}",
        f"                            -> {s['classification']}",
        "            solute MASS:    "
        f"S(k->0) = {sm['S_low']:.3f}, alpha = {sm['alpha']:+.2f} "
        f"+/- {sm['alpha_se']:.2f}",
        f"                            -> {sm['classification']}",
        "            concentration FIELD:  "
        f"alpha_c = {stage['alpha_field']:+.2f} +/- {stage['alpha_field_se']:.2f}",
        "",
        "  READ IT   What the conserved dynamics organise is the SOLUTE, not "
        "the domain count.",
        "            Long-wavelength composition fluctuations cannot be built "
        "without transport",
        "            over that wavelength, so the field is suppressed at small "
        "k; how many domains",
        "            hold the local solute budget is left free, and that "
        "freedom is what the",
        "            unweighted centroid spectrum measures.",
        f"            Lowest accessible wavevector 2pi/L = {2 * np.pi / L:.4f} "
        "cell^-1: over so short a",
        "            range an exponent is a description of the accessible "
        "window, not a class.",
        "  " + "-" * 72,
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
