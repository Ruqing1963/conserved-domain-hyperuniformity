# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------
# VENDORED FILE -- not developed for this repository.
#
# Source: the phase-field solver of the companion study on kinetic arrest
# in sphalerite (repository `sphalerite-kinetic-arrest`), copied here
# unmodified so that this repository can be cloned and run on its own.
#
# The alternative -- a submodule, or an import from a sibling checkout --
# reproduces more elegantly and works less often: anyone who forgets
# `--recursive` gets an ImportError instead of a result. Duplication is the
# cheaper failure mode for a reproducibility archive.
#
# If the upstream solver changes, this copy does NOT track it. Record the
# upstream commit hash here when publishing:
#     upstream commit: <fill in before release>
# ---------------------------------------------------------------------

"""
cooling_2d.py  --  Paper 3, Section 5

A complete, runnable two-dimensional Cahn-Hilliard simulation of a cooling
sphalerite grain, producing the morphological sequence

    homogeneous  ->  spinodal decomposition  ->  coarsening  ->  arrest

and saving the concentration field at each stage.

    python cooling_2d.py                 # default: three cooling rates
    python cooling_2d.py --rate 3e-5     # a single rate
    python cooling_2d.py --quick         # small grid, for a fast check

WHAT IS BEING VARIED

Only one dimensionless quantity matters, the ratio of the cooling timescale to
the diffusive timescale. Cooling enters through an Arrhenius mobility: as T
falls, M(T) = M0 exp(-Ea/kB T) collapses, and coarsening -- which is driven by
curvature and limited by transport -- freezes. Fast cooling arrests the
microstructure early, leaving many small domains; slow cooling lets it run to
completion, leaving one. That competition is the physical content of the paper,
and it is what these runs display.

WHAT THE PICTURES DO AND DO NOT SHOW

They show the PATHWAY and the TOPOLOGY: which morphologies the system passes
through, whether coarsening completes or arrests, and how that depends on the
cooling rate. Those statements are dimensionless and are unaffected by the
numerical widening of the interface.

They do NOT show the absolute domain size. The characteristic length is
sqrt(kappa/|f''|); with kappa fixed from an atomically sharp interface it is of
order one lattice parameter, which no grid resolves, so kappa is inflated until
the interface spans four to six cells. Domain sizes here are therefore set by
the grid. Absolute sizes must come from direct lattice-model coarsening, where
the interface is what it physically is. Inflating kappa rescales lengths and
times together, so the dimensionless conclusions survive; the length axis of
every figure produced here is labelled in grid units for that reason.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

KB_EV = 8.617333262e-5
_HERE = os.path.dirname(os.path.abspath(__file__))
_OUT = os.path.join(_HERE, os.pardir, "figures")
_DATA = os.path.join(_HERE, os.pardir, "data")


# ======================================================================
# 1. Free energy
# ======================================================================
class SphaleriteFreeEnergy:
    """
    Bulk free energy of the lattice model, in a form that can safely drive the
    dynamics.

    WHY NOT SIMPLY A POLYNOMIAL FIT

    The obvious step is to fit a quartic to the eight measured points and use
    its derivative. That fails, and the failure is instructive. A least-squares
    quartic through data spanning c = 0.005 to 0.20 comes out with a NEGATIVE
    leading coefficient, so f -> -infinity as c grows: at c = 1 the fitted f is
    -22.9 eV per site. Nothing in the dynamics keeps c inside the fitted window,
    and once a fluctuation carries a cell outside it the free energy rewards
    going further, so the field diverges within a few hundred steps. A
    polynomial fitted to a narrow window cannot serve as a global free energy.

    WHAT IS USED INSTEAD

    The measured points are embedded in a double well that is bounded below and
    grows without limit outside [0, 1]:

        f(c) = W c^2 (1-c)^2 + g(c),

    with W chosen so that the well dominates far from the measured range, and
    g(c) a low-order correction fitted to reproduce the measured points inside
    it. The double well supplies global stability and the two-phase structure;
    the correction supplies the measured shape where it was measured.

    This is a numerical device, not a physical claim. The manuscript diagnoses
    the instability by the common-tangent construction on the raw measurements,
    which requires neither fit nor differentiation. The role of f(c) here is to
    drive the dynamics with the right qualitative structure -- two coexisting
    compositions separated by a barrier -- and the arrest physics the paper
    concerns does not depend on the fine shape of the well.
    """

    C_MEAS = np.array([0.005, 0.010, 0.020, 0.040, 0.080, 0.120, 0.160, 0.200])
    F_MEAS = np.array([0.00059, 0.00089, 0.00101, 0.00083,
                       -0.00214, -0.00258, -0.00405, -0.00513])

    def __init__(self, scale=100.0, W=None, c_beta=0.5, degree=2):
        """
        scale multiplies f. It is a numerical convenience: the measured free
        energies are of order 1e-3 eV, which with a small mobility makes for
        very stiff stepping. Only the product M f'' sets the rate, so scaling f
        and 1/M together leaves the dynamics unchanged.

        c_beta is the composition of the solute-rich phase. The lattice model
        condenses to a domain whose interior approaches the roquesite ordering,
        i.e. c -> 1 on the cation sublattice; 0.5 is used here so that both
        phases are resolved on a modest grid.
        """
        self.scale = float(scale)
        self.c_beta = float(c_beta)
        # depth chosen so the well dominates the measured correction by ~10x
        self.W = float(W) if W is not None else 40.0 * abs(self.F_MEAS).max()
        # correction fitted to the residual inside the measured window
        well = self._well(self.C_MEAS)
        self.coef = np.polyfit(self.C_MEAS, self.F_MEAS - well, degree)
        self._g = np.poly1d(self.coef)
        self._dg = np.polyder(self._g, 1)
        self._d2g = np.polyder(self._g, 2)
        # taper the correction to zero outside the measured window so that the
        # bounded double well alone governs the far field
        self.c_lo, self.c_hi = self.C_MEAS.min(), self.C_MEAS.max()
        for cc in (0.05, 0.15, 0.30, 0.45):
            a, b, d, e = self.check_derivatives(cc)
            assert abs(a - b) < 1e-2 * max(1.0, abs(b)), \
                f"dfdc inconsistent with f_of_c at c={cc}: {a:.4f} vs {b:.4f}"
            assert abs(d - e) < 5e-2 * max(1.0, abs(e)), \
                f"d2fdc2 inconsistent at c={cc}: {d:.2f} vs {e:.2f}"

    def _well(self, c):
        b = self.c_beta
        return self.W * (c / b) ** 2 * (1 - c / b) ** 2

    def _dwell(self, c):
        b = self.c_beta
        u = c / b
        return self.W * 2 * u * (1 - u) * (1 - 2 * u) / b

    def _d2well(self, c):
        b = self.c_beta
        u = c / b
        return self.W * 2 * (1 - 6 * u + 6 * u ** 2) / b ** 2

    def _taper(self, c):
        """
        Weight that is 1 inside the measured window and falls smoothly to 0
        outside it, so that beyond the range where f(c) was measured the
        bounded double well alone governs the free energy.
        """
        w = self.c_hi - self.c_lo
        u = (np.clip(c, -1.0, 2.0) - 0.5 * (self.c_lo + self.c_hi)) / (0.9 * w)
        return np.exp(-u ** 4)

    def _dtaper(self, c):
        """
        Derivative of the taper.

        This term must NOT be dropped. An earlier version omitted it on the
        grounds that it "affects only the far field", which is wrong: the taper
        varies fastest precisely where the correction hands over to the well,
        and at c = 0.30 leaving it out made df/dc inconsistent with d2f/dc2 by a
        factor of 150. The dynamics then had no relation to the free energy that
        the second derivative described, and the field decayed where linear
        stability said it should grow.
        """
        w = self.c_hi - self.c_lo
        s = 0.9 * w
        u = (np.clip(c, -1.0, 2.0) - 0.5 * (self.c_lo + self.c_hi)) / s
        return -4.0 * u ** 3 * np.exp(-u ** 4) / s

    def _d2taper(self, c):
        w = self.c_hi - self.c_lo
        s = 0.9 * w
        u = (np.clip(c, -1.0, 2.0) - 0.5 * (self.c_lo + self.c_hi)) / s
        return (16.0 * u ** 6 - 12.0 * u ** 2) * np.exp(-u ** 4) / s ** 2

    def f_of_c(self, c):
        return self.scale * (self._well(c) + self._taper(c) * self._g(c))

    def dfdc(self, c):
        """
        Derivative of f. The taper derivative must be included: it varies
        fastest exactly where the fitted correction hands over to the well, and
        omitting it made df/dc inconsistent with d2f/dc2 by a factor of 150 at
        c = 0.30. The dynamics then bore no relation to the free energy the
        second derivative described, and the field decayed where linear
        stability said it should grow.
        """
        return self.scale * (self._dwell(c)
                             + self._dtaper(c) * self._g(c)
                             + self._taper(c) * self._dg(c))

    def d2fdc2(self, c):
        return self.scale * (self._d2well(c)
                             + self._d2taper(c) * self._g(c)
                             + 2.0 * self._dtaper(c) * self._dg(c)
                             + self._taper(c) * self._d2g(c))

    def check_derivatives(self, c=0.30, h=1e-5):
        """
        Finite-difference check that dfdc and d2fdc2 really are the derivatives
        of f_of_c. Asserted at construction, because an inconsistency here is
        silent: the solver runs, produces plausible pictures, and describes a
        free energy that is not the one it reports.
        """
        fd1 = (self.f_of_c(c + h) - self.f_of_c(c - h)) / (2 * h)
        fd2 = (self.f_of_c(c + h) - 2 * self.f_of_c(c)
               + self.f_of_c(c - h)) / h ** 2
        return float(self.dfdc(c)), float(fd1), float(self.d2fdc2(c)), float(fd2)

    def unstable_range(self, lo=0.0, hi=None, n=600):
        hi = self.c_beta if hi is None else hi
        cc = np.linspace(lo, hi, n)
        neg = cc[self.d2fdc2(cc) < 0]
        return (float(neg.min()), float(neg.max())) if neg.size else None

    def fit_quality(self):
        pred = (self._well(self.C_MEAS)
                + self._taper(self.C_MEAS) * self._g(self.C_MEAS))
        return float(np.sqrt(np.mean((pred - self.F_MEAS) ** 2)))

    def is_bounded(self, lo=-0.5, hi=1.5, n=400):
        """Sanity check: f must not run away outside the physical range."""
        cc = np.linspace(lo, hi, n)
        return bool(np.all(np.isfinite(self.f_of_c(cc)))
                    and self.f_of_c(cc).min() > -1e3 * self.scale)


# ======================================================================
# 2. Solver
# ======================================================================
class CahnHilliard2D:
    """
    Semi-implicit spectral integrator on a periodic square grid.

    The fourth-order linear term is diagonal in Fourier space and is treated
    implicitly; the nonlinear bulk term explicitly:

        c_hat <- (c_hat - dt M k^2 F[df/dc]) / (1 + dt M kappa k^4)

    An explicit scheme would need dt < C dx^4 / (M kappa), which is hopeless at
    these resolutions.
    """

    def __init__(self, n, dx, kappa, free, seed=0):
        self.n, self.dx, self.kappa, self.free = int(n), float(dx), float(kappa), free
        kx = 2 * np.pi * np.fft.fftfreq(self.n, d=self.dx)
        ky = 2 * np.pi * np.fft.rfftfreq(self.n, d=self.dx)
        KX, KY = np.meshgrid(kx, ky, indexing="ij")
        self.k2 = KX ** 2 + KY ** 2
        self.k4 = self.k2 ** 2
        self.rng = np.random.default_rng(seed)
        self.c = None

    def initialise(self, c0, noise=0.002):
        self.c = c0 + noise * (self.rng.random((self.n, self.n)) - 0.5)
        self.c0 = float(c0)
        return self.c

    def step(self, dt, M):
        c_hat = np.fft.rfft2(self.c)
        mu_hat = np.fft.rfft2(self.free.dfdc(self.c))
        self.c = np.fft.irfft2(
            (c_hat - dt * M * self.k2 * mu_hat) / (1.0 + dt * M * self.kappa * self.k4),
            s=(self.n, self.n))
        return self.c

    # ---------------- diagnostics ----------------
    def domain_scale(self):
        """
        Characteristic length from the first moment of the structure factor,
        L = 2 pi <S> / <k S>. Far less noisy than counting domains, and the
        standard measure in the coarsening literature.
        """
        d = self.c - self.c.mean()
        S = np.abs(np.fft.rfft2(d)) ** 2
        k = np.sqrt(self.k2)
        num, den = float(np.sum(k * S)), float(np.sum(S))
        return float(2 * np.pi * den / num) if num > 0 else np.nan

    def free_energy(self):
        gx = (np.roll(self.c, -1, 0) - np.roll(self.c, 1, 0)) / (2 * self.dx)
        gy = (np.roll(self.c, -1, 1) - np.roll(self.c, 1, 1)) / (2 * self.dx)
        return float(np.mean(self.free.f_of_c(self.c)
                             + 0.5 * self.kappa * (gx ** 2 + gy ** 2)))

    def amplitude(self):
        """Standard deviation of the field: near zero while homogeneous, large
        once decomposition has occurred. Distinguishes stage 1 from stage 2."""
        return float(self.c.std())

    def n_domains(self):
        """Connected regions above the mean, by flood fill on a boolean mask."""
        mask = self.c > self.c.mean()
        seen = np.zeros_like(mask, dtype=bool)
        n = 0
        idx = np.argwhere(mask & ~seen)
        for i0, j0 in idx:
            if seen[i0, j0]:
                continue
            n += 1
            stack = [(i0, j0)]
            seen[i0, j0] = True
            while stack:
                i, j = stack.pop()
                for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    a, b = (i + di) % self.n, (j + dj) % self.n
                    if mask[a, b] and not seen[a, b]:
                        seen[a, b] = True
                        stack.append((a, b))
        return n


# ======================================================================
# 3. Cooling protocol
# ======================================================================
def mobility(T, Ea=0.60, M0=1.0):
    """
    M(T) = M0 exp(-Ea / kB T).

    Ea has a lower bound from the model: removing a solute from a domain breaks
    heterovalent bonds costing 2*lambda ~ 0.6 eV each. That is a bound, not a
    determination -- the true barrier also contains the migration barrier of the
    diffusion mechanism, which the configurational model does not represent. M0
    is unity, so time is measured in units of the diffusive time at infinite
    temperature and every result is a RELATIVE calibration.
    """
    return M0 * np.exp(-Ea / (KB_EV * T))


def run_cooling(n_cool_steps, n=256, dx=1.0, kappa=None, c0=0.30, T_hi=1800.0,
                T_lo=500.0, seed=0, f_scale=100.0, lam_target=12.0,
                Ea=0.60, target_amp=1.02, n_snapshots=6, verbose=True):
    """
    Cool from T_hi to T_lo, recording snapshots along the way.

    HOW COOLING IS PARAMETERISED, AND WHY NOT BY PHYSICAL TIME

    The mobility spans four orders of magnitude between T_hi and T_lo, so no
    single timestep serves both ends: one small enough to be stable while hot
    wastes millions of steps once cold, where nothing moves. The step is
    therefore adaptive,

        dt = cfl / ( M |f''|_max k_max^2 ),

    growing as M collapses.

    The cooling schedule is then specified in STEPS rather than in physical
    time, T descending linearly from T_hi to T_lo over n_cool_steps. This is
    deliberate. What controls the morphology is the ratio of the cooling rate to
    the coarsening rate, and because dt already tracks the mobility, a fixed
    number of steps per kelvin is a fixed amount of DIFFUSIVE PROGRESS per
    kelvin -- which is exactly the dimensionless ratio the paper is about.
    Specifying the schedule in physical time instead would make the fast-cooling
    runs terminate before the mobility had fallen at all, which is a statement
    about the units chosen rather than about the physics.

    A LARGE n_cool_steps therefore means SLOW cooling (much diffusion per
    kelvin) and a small one means FAST cooling. Physical time is accumulated
    alongside and reported.

    COMPOSITION. c0 = 0.30 places the system inside the unstable region of the
    fitted free energy so that the SPINODAL pathway is displayed. This is not
    the natural indium content: the measured f(c) spans 0.005 to 0.20 and
    natural tenors are far more dilute still, where decomposition would proceed
    by nucleation, which this solver does not treat. The arrest physics that the
    paper concerns is the same either way.
    """
    free = SphaleriteFreeEnergy(scale=f_scale)

    # kappa is chosen so that the fastest-growing spinodal wavelength spans a
    # resolvable number of cells. This is the numerical widening announced in
    # Section 2.5, here made explicit and quantitative.
    #
    # Linear stability of the Cahn-Hilliard equation gives a fastest-growing
    # wavevector k_max^2 = -f''/(2 kappa), hence a wavelength
    # lambda = 2 pi sqrt(2 kappa / |f''|). With kappa at its physical value the
    # interface is one bond wide, lambda is of order the lattice parameter, and
    # a grid that resolved it would span a few nanometres in total. We therefore
    # set kappa from the target wavelength instead:
    #
    #     kappa = |f''| lambda_target^2 / (8 pi^2).
    #
    # Inflating kappa rescales lengths and times together, so the dimensionless
    # statements this paper rests on -- the ORDER of morphologies and whether
    # coarsening arrests -- are untouched. What is lost is the absolute scale,
    # which comes instead from direct lattice-model coarsening.
    if kappa is None:
        d2_at_c0 = abs(free.d2fdc2(c0))
        kappa = d2_at_c0 * lam_target ** 2 / (8 * np.pi ** 2)
    sim = CahnHilliard2D(n, dx, kappa, free, seed=seed)
    sim.initialise(c0, noise=0.002)

    d2max = max(abs(free.d2fdc2(x)) for x in np.linspace(0.0, 0.6, 200))
    k2max = (np.pi / dx) ** 2 * 2          # 2D corner of the Brillouin zone

    # ---- timestep ----
    #
    # Linearising the scheme about c0, a mode k is amplified per step by
    #
    #     A(k) = (1 - dt M k^2 f'') / (1 + dt M kappa k^4),
    #
    # so growth requires -f'' > kappa k^2, which is the linear stability
    # condition and does not involve dt. What dt controls is the SIZE of the
    # growth per step.
    #
    # An earlier version set dt from an explicit-scheme CFL condition,
    # dt = 0.25 / (M |f''|_max k_max^2). That is stable, but it makes
    # A(k) - 1 ~ 3e-4, so the unstable mode needs some 7000 steps to grow
    # tenfold while the run has only a few thousand at that mobility. The
    # result looked like diffusive smoothing rather than decomposition,
    # because that is what it was.
    #
    # The whole point of treating the stiff fourth-order term implicitly is
    # that it tolerates a far larger dt. We therefore set dt from the target
    # amplification of the fastest-growing mode, and report the achieved value
    # so that the choice is auditable rather than tuned by eye.
    d2_c0 = free.d2fdc2(c0)
    if d2_c0 < 0:
        kk = np.linspace(1e-3, np.pi / dx, 800)
        k_grow = kk[np.argmax(-kk ** 2 * (d2_c0 + kappa * kk ** 2))]
        # solve A = target for dt at k_grow
        num = target_amp - 1.0
        den = -k_grow ** 2 * d2_c0 - target_amp * kappa * k_grow ** 4
        dt_unit = num / den if den > 0 else 1.0 / (d2max * k2max)
    else:
        k_grow = np.nan
        dt_unit = 1.0 / (d2max * k2max)

    T, t, step = float(T_hi), 0.0, 0
    dT_total = T_hi - T_lo
    snapshots, trace = [], {k: [] for k in ("t", "T", "M", "dt", "L", "F", "amp", "nd")}
    snap_T = np.linspace(T_hi, T_lo, n_snapshots)
    next_snap = 1

    if verbose:
        lam = 2 * np.pi * np.sqrt(2 * kappa / max(abs(free.d2fdc2(c0)), 1e-12))
        print(f"  kappa = {kappa:.2f} -> spinodal wavelength "
              f"{lam:.1f} cells ({n/lam:.0f} domains across the box)")
        if np.isfinite(k_grow):
            A = ((1 - dt_unit * k_grow ** 2 * d2_c0)
                 / (1 + dt_unit * kappa * k_grow ** 4))
            print(f"  dt set for amplification {A:.4f} per step at the "
                  f"fastest-growing mode")
            print(f"    -> {np.log(10)/np.log(A):.0f} steps for a tenfold "
                  f"growth of the composition contrast")
        print(f"  {n_cool_steps:,} steps over {dT_total:.0f} K "
              f"({'slow' if n_cool_steps > 20000 else 'fast'} cooling), "
              f"adaptive dt")

    t0 = time.time()
    snapshots.append(dict(step=0, T=T, t=0.0, M=mobility(T, Ea=Ea),
                          c=sim.c.copy(), L=sim.domain_scale(),
                          amp=sim.amplitude()))
    for step in range(1, n_cool_steps + 1):
        T = T_hi - dT_total * step / n_cool_steps
        M = mobility(T, Ea=Ea)
        dt = dt_unit / max(M, 1e-30)
        sim.step(dt, M)
        t += dt

        if step % max(1, n_cool_steps // 200) == 0:
            trace["t"].append(t); trace["T"].append(T); trace["M"].append(M)
            trace["dt"].append(dt)
            trace["L"].append(sim.domain_scale())
            trace["F"].append(sim.free_energy())
            trace["amp"].append(sim.amplitude())
            trace["nd"].append(sim.n_domains() if n <= 192 else -1)
        if next_snap < len(snap_T) and T <= snap_T[next_snap]:
            snapshots.append(dict(step=step, T=float(T), t=float(t), M=float(M),
                                  c=sim.c.copy(), L=sim.domain_scale(),
                                  amp=sim.amplitude()))
            next_snap += 1
        if not np.isfinite(sim.c).all():
            raise RuntimeError(f"field diverged at step {step}, T = {T:.0f} K")

    if len(snapshots) < n_snapshots:
        snapshots.append(dict(step=step, T=float(T), t=float(t),
                              M=float(mobility(T, Ea=Ea)), c=sim.c.copy(),
                              L=sim.domain_scale(), amp=sim.amplitude()))
    if verbose:
        print(f"    {step:,} steps, {time.time()-t0:.0f} s; final L = "
              f"{sim.domain_scale():.1f} grid units, "
              f"{sim.n_domains()} domains")
    return sim, {k: np.asarray(v) for k, v in trace.items()}, snapshots


# ======================================================================
# 4. Figures
# ======================================================================
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8.5, "pdf.fonttype": 42,
                         "ps.fonttype": 42})
    return plt


def plot_sequence(snapshots, n_cool_steps, outfile):
    """
    The morphological sequence for one cooling schedule.

    Each panel is scaled to its own range rather than to a common one. The
    initial condition is a small-amplitude random field whose extremes are set
    by the seeding noise, not by the decomposition; putting it on a shared
    colour scale compresses every later panel into a narrow band and hides the
    morphology the figure exists to show. The amplitude is instead reported
    numerically under each panel, so the growth of the composition contrast
    remains visible.
    """
    plt = _mpl()
    m = len(snapshots)
    fig, axes = plt.subplots(1, m, figsize=(2.0 * m, 2.8))
    for a, s in zip(np.atleast_1d(axes), snapshots):
        c = s["c"]
        a.imshow(c, cmap="magma", vmin=c.min(), vmax=c.max(),
                 interpolation="nearest")
        a.set_xticks([])
        a.set_yticks([])
        a.set_title("T = {:.0f} K\nL = {:.0f}, dc = {:.3f}".format(
            s["T"], s["L"], c.max() - c.min()), fontsize=8)
    fig.suptitle("Cooling over {:,} steps. Panels are individually scaled; "
                 "lengths are in GRID UNITS (Section 2.5).".format(n_cool_steps),
                 fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    for ext in ("pdf", "png"):
        fig.savefig(outfile + "." + ext, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_schedule_comparison(results, outfile):
    """Final microstructures and coarsening histories across cooling speeds."""
    plt = _mpl()
    keys = sorted(results)
    fig = plt.figure(figsize=(3.0 * len(keys), 5.8))
    gs = fig.add_gridspec(2, len(keys), height_ratios=[1.25, 1.0])

    for i, ns in enumerate(keys):
        sim, tr, snaps = results[ns]
        a = fig.add_subplot(gs[0, i])
        a.imshow(snaps[-1]["c"], cmap="magma", interpolation="nearest")
        a.set_xticks([])
        a.set_yticks([])
        a.set_title("{:,} steps\nL = {:.0f}, {:d} domains".format(
            ns, snaps[-1]["L"], sim.n_domains()), fontsize=8.5)

    b = fig.add_subplot(gs[1, :])
    for ns in keys:
        _, tr, _ = results[ns]
        ok = np.isfinite(tr["L"])
        b.plot(tr["T"][ok], tr["L"][ok], "-", lw=1.4,
               label="{:,} steps".format(ns))
    b.invert_xaxis()
    b.set_xlabel("temperature (K), decreasing to the right")
    b.set_ylabel("domain scale L (grid units)")
    b.legend(fontsize=7.5)
    b.grid(alpha=0.25, lw=0.5)
    b.set_title("Coarsening runs further, and to a larger scale, the slower "
                "the cooling", fontsize=9, loc="left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(outfile + "." + ext, bbox_inches="tight", dpi=150)
    plt.close(fig)


# ======================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=None,
                    help="cooling steps for a single run; larger = slower")
    ap.add_argument("--quick", action="store_true", help="small grid, fast")
    ap.add_argument("--n", type=int, default=None, help="grid size")
    args = ap.parse_args()

    os.makedirs(_OUT, exist_ok=True)
    os.makedirs(_DATA, exist_ok=True)

    n = args.n if args.n else (96 if args.quick else 192)
    free = SphaleriteFreeEnergy(scale=100.0)
    print("Cahn-Hilliard cooling simulation, 2D")
    print("  free energy: bounded double well plus a correction fitted to the")
    print("  eight thermodynamic-integration points of Section 2.2")
    print("    fit RMS      {:.2e} eV per site".format(free.fit_quality()))
    print("    bounded      {}".format(free.is_bounded()))
    print("    unstable over c = {}".format(
        tuple(round(x, 3) for x in free.unstable_range())))
    print("  grid {} x {}\n".format(n, n))

    if args.steps:
        schedules = [args.steps]
    elif args.quick:
        schedules = [3000]
    else:
        schedules = [2000, 8000, 32000]

    results = {}
    for ns in schedules:
        sim, tr, snaps = run_cooling(ns, n=n, seed=7)
        results[ns] = (sim, tr, snaps)
        plot_sequence(snaps, ns, os.path.join(_OUT, "fig_sequence_%d" % ns))
        np.savez_compressed(os.path.join(_DATA, "cooling_%d.npz" % ns),
                            c_final=snaps[-1]["c"], **tr)

    if len(results) > 1:
        plot_schedule_comparison(results, os.path.join(_OUT, "fig_cooling_rates"))
        print("\n{:>8} {:>8} {:>9} {:>9}".format(
            "steps", "speed", "L final", "domains"))
        summary = {}
        for ns in sorted(results):
            sim, tr, snaps = results[ns]
            nd = sim.n_domains()
            speed = "fast" if ns <= 4000 else ("slow" if ns >= 20000 else "medium")
            print("{:8d} {:>8} {:9.1f} {:9d}".format(
                ns, speed, snaps[-1]["L"], nd))
            summary[str(ns)] = dict(cool_steps=ns,
                                    L_final=float(snaps[-1]["L"]),
                                    n_domains=int(nd))
        json.dump(summary,
                  open(os.path.join(_DATA, "cooling_summary.json"), "w"),
                  indent=2)
        nss = np.array([summary[k]["cool_steps"] for k in summary], float)
        Ls = np.array([summary[k]["L_final"] for k in summary], float)
        if np.all(Ls > 0) and len(Ls) > 2:
            expo = np.polyfit(np.log(nss), np.log(Ls), 1)[0]
            print("\n  final domain scale versus cooling slowness: "
                  "L ~ steps^{:+.3f}".format(expo))
            print("  A positive exponent is the signature of arrest: slower")
            print("  cooling lets coarsening proceed further before the")
            print("  mobility collapses, leaving fewer and larger domains.")

    print("\nFigures written to {}".format(_OUT))
