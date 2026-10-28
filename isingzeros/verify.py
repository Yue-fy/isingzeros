"""
Verification suite (run on a CPU server; see README.md for the meaning of every test,
expected outputs and pass/fail criteria).

    python -m isingzeros.verify v1   [--quick]          # Fejer-Riesz structure: exactness, symmetry, sizes, ladders
    python -m isingzeros.verify v2   [--quick]          # shot noise: bias / rms / bootstrap coverage (L=6, M=3)
    python -m isingzeros.verify v3   [--quick]          # circuit-level noise (density matrix): Pauli invariance, fit designs, floor modes, ZNE
    python -m isingzeros.verify v4   [--quick]          # sign of a0: two-point post-selected test, ideal and noisy
    python -m isingzeros.verify v5   [--quick]          # measurability window: root error vs |a0| over sizes
    python -m isingzeros.verify v6   [--quick]          # Fejer-Riesz vs LCU 2D localisation at equal noise
    python -m isingzeros.verify v7                      # reset redundancy under post-selection
    python -m isingzeros.verify v8   [--quick]          # coverage of the LCU bootstrap regions
    python -m isingzeros.verify report                  # collect results/verify/*.json -> results/verify/REPORT.md
Common options: --nproc N (process pool), --seed S, --outdir results/verify
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
from . import fejer as fr

OUT = 'results/verify'


# ----------------------------------------------------------------------------- helpers
def pmap(f, args, nproc):
    if nproc <= 1:
        return [f(x) for x in args]
    from multiprocessing import Pool
    with Pool(nproc) as p:
        return p.map(f, args)


def _pA(args):
    """Noisy protocol-A return probability (density matrix)."""
    L, M, phi, th, p2, p1, pro = args
    from .analysis import noisy_joint_probability_dm
    from .circuits import kicked_ising_unitary
    if p2 == 0 and p1 == 0 and pro == 0:
        return abs(fr.amplitude(L, M, phi, 1j * th)) ** 2
    return noisy_joint_probability_dm(kicked_ising_unitary(L, M, th, phi), p2, p1, pro)


def _pB(args):
    """Noisy LCU joint probability (density matrix)."""
    L, M, phi, a, th, p2, p1, pro, reset = args
    from .analysis import noisy_joint_probability_dm, joint_probability
    from .circuits import kicked_ising_lcu
    if p2 == 0 and p1 == 0 and pro == 0 and reset:
        return joint_probability(L, M, phi, a, th)
    return noisy_joint_probability_dm(kicked_ising_lcu(L, M, th, phi, a, 0.0, reset=reset), p2, p1, pro)


def save(name, obj, outdir):
    os.makedirs(outdir, exist_ok=True)
    f = os.path.join(outdir, name)
    json.dump(obj, open(f, 'w'), indent=1, default=lambda x: (x.real, x.imag) if isinstance(x, complex) else str(x))
    print('->', f)


def circ_diff(t1, t2):
    return float(np.angle(np.exp(4j * (t1 - t2))) / 4)


# ----------------------------------------------------------------------------- V1
def v1(a):
    """Noiseless: zeros from |A|^2 on the axis (Fejer-Riesz) == zeros from the complex polynomial == transfer-
    matrix grid search; symmetry (a, theta) -> (-a, pi/2 - theta); on-axis zeros; ladders."""
    from .analysis import exact_zeros_K2plane
    sizes = [(4, 2, 1), (4, 3, 1), (6, 2, 1), (6, 3, 1), (6, 4, 1), (8, 2, 1), (8, 3, 1), (8, 4, 1), (10, 3, 1), (10, 4, 1), (12, 3, 1), (4, 2, 2), (4, 3, 2)]
    if a.quick:
        sizes = [(4, 2, 1), (6, 3, 1), (8, 3, 1), (8, 4, 1), (4, 2, 2)]
    phis = [0.4, 0.6, np.pi / 4]
    rows = []
    for (L, M, W) in sizes:
        for phi in phis:
            t0 = time.time()
            th3 = 0.3 if W > 1 else 0.0
            n = fr.degree(L, M, W)
            zs = fr.exact_zeros(L, M, phi, W, th3)
            th = fr.theta_grid(n)
            P = fr.return_probabilities_exact(L, M, phi, th, W, th3)
            pairs, r, B = fr.reconstruct(th, P, n)
            m = fr.match(pairs, zs)
            err = max((max(abs(x['err_a']), abs(x['err_theta'])) for x in m), default=0.0)
            # multiplicity of the most degenerate root (multiple roots are resolved only to ~eps^(1/mult))
            mult = max((sum(abs(w - z) < 1e-3 for w in zs) for z in zs), default=0)
            # symmetry check
            sym = max(min(np.hypot(-z.real - w.real, circ_diff(np.pi / 2 - z.imag, w.imag)) for w in zs) for z in zs) if len(zs) else 0.0
            onaxis = [float(z.imag) for z in zs if abs(z.real) < 1e-5]
            grid_err = None
            if L * W <= 8 and not a.quick:
                zg = exact_zeros_K2plane(L, M, phi, W=W, theta3=th3, a_max=0.25)
                grid_err = max((min(abs(g - z) for z in zs) for g in zg), default=0.0)
            rows.append(dict(L=L, M=M, W=W, phi=phi, degree=n, n_zeros=len(zs), max_multiplicity=mult, max_err_FR_vs_exact=err,
                             symmetry_residual=sym, onaxis_zero_thetas=onaxis, grid_vs_poly=grid_err,
                             zeros=[[z.real, z.imag] for z in zs], seconds=time.time() - t0))
            print(f'L={L:2d} M={M} W={W} phi={phi:.3f} n={n:2d} ({len(zs)} finite zeros, max multiplicity {mult}): max |FR - exact| = {err:.1e}  symmetry {sym:.1e}  '
                  f'on-axis zeros at {np.round(onaxis, 4).tolist()}' + (f'  grid-vs-poly {grid_err:.1e}' if grid_err is not None else ''))
    save('v1_structure.json', dict(rows=rows), a.outdir)


# ----------------------------------------------------------------------------- V2
def v2(a):
    """Shot noise only, L=6 M=3 phi=0.6: bias / rms of (|a|, theta) for every zero vs K and S; coverage of the
    parametric-bootstrap 95 % intervals."""
    L, M, phi = 6, 3, 0.6
    n = fr.degree(L, M); zs = fr.exact_zeros(L, M, phi)
    rng = np.random.default_rng(a.seed)
    Ks = [20, 40, 80] if not a.quick else [40]
    Ss = [1000, 10000, 50000, 200000] if not a.quick else [10000, 50000]
    R = 200 if not a.quick else 60
    rows = []
    for K in Ks:
        th = fr.theta_grid(n, K)
        P = fr.return_probabilities_exact(L, M, phi, th)
        for S in Ss:
            errs = []
            for _ in range(R):
                pairs, *_ = fr.sample_and_reconstruct(th, P, S, n, rng, symmetric=True)
                errs.append([(x['err_a'], x['err_theta']) for x in fr.match(pairs, zs)])
            errs = np.array(errs)
            for i, z in enumerate(zs):
                e = errs[:, i, :]
                rows.append(dict(K=K, S=S, total_shots=K * S, a0=z.real, theta0=z.imag, bias_a=float(e[:, 0].mean()),
                                 rms_a=float(np.sqrt((e[:, 0] ** 2).mean())), bias_theta=float(e[:, 1].mean()),
                                 rms_theta=float(np.sqrt((e[:, 1] ** 2).mean()))))
            near = [r for r in rows[-len(zs):] if abs(abs(r['a0']) - 0.0213) < 1e-3]
            print(f'K={K:3d} S={S:6d} (total {K*S:.1e}): near-axis zeros |a0|=0.0213 -> rms |a| {near[0]["rms_a"]:.4f}, rms theta {near[0]["rms_theta"]:.4f}; '
                  f'max rms over all zeros {max(r["rms_a"] for r in rows[-len(zs):]):.4f}')
    # coverage of bootstrap intervals at K=40, S=5e4
    K, S = 40, 50000
    th = fr.theta_grid(n, K); P = fr.return_probabilities_exact(L, M, phi, th)
    Rc = 100 if not a.quick else 20; nb = 100 if not a.quick else 40
    hit = np.zeros((len(zs), 2)); width = []
    for _ in range(Rc):
        k = rng.binomial(S, np.clip(P, 0, 1))
        pairs0, smp = fr.bootstrap(th, k, S, n, nboot=nb, rng=rng, symmetric=True)
        m = fr.match(pairs0, zs)
        for i, mm in enumerate(m):
            j = mm['index']
            lo_a, hi_a = np.percentile(smp[:, j, 0], [2.5, 97.5]); lo_t, hi_t = np.percentile(smp[:, j, 1], [2.5, 97.5])
            hit[i, 0] += lo_a - 1e-6 <= abs(zs[i].real) <= hi_a + 1e-6
            hit[i, 1] += lo_t <= zs[i].imag <= hi_t
        width.append([np.ptp(np.percentile(smp[:, j, 0], [2.5, 97.5])) for j in range(len(pairs0))])
    cov = (hit / Rc).tolist()
    print(f'bootstrap 95% coverage at K={K}, S={S} over {Rc} datasets (|a|, theta) per zero: ' + ', '.join(f'({c[0]:.2f},{c[1]:.2f})' for c in cov))
    save('v2_shotnoise.json', dict(L=L, M=M, phi=phi, rows=rows, coverage=dict(K=K, S=S, R=Rc, nboot=nb, per_zero=cov,
                                                                               zeros=[[z.real, z.imag] for z in zs])), a.outdir)


# ----------------------------------------------------------------------------- V3
# One theta grid (40 points on one period [0, pi/2), shifted by half a step; 5e4 shots each = 2e6 shots) and two fits:
#   'sym': cosine part only (n + 1 coefficients), sine part discarded and reported (non-Pauli diagnostic)
#   'gen': all 2n + 1 allowed coefficients (the revision-4 'half' design)
# Under any Pauli noise channel the data contain no sine part, so 'sym' loses nothing and has fewer parameters.
GRIDS = {'sym': dict(K=40, period=np.pi / 2, symmetric=True, S=50000),
         'gen': dict(K=40, period=np.pi / 2, symmetric=False, S=50000)}


def fr_grid(n, name):
    g = GRIDS[name]
    return fr.theta_grid(n, g['K'], period=g['period']), g


def v3(a):
    """Circuit-level noise (density matrix, angle-independent compilation): Pauli invariance of the data,
    deterministic position bias (infinite shots) and finite-shot rms for the two fits and three floor modes,
    size of the discarded sine part, and zero-noise extrapolation of the positions (gate noise x1, x2, x3)."""
    L, M, phi = 6, 3, 0.6
    n = fr.degree(L, M); zs = fr.exact_zeros(L, M, phi)
    th, _ = fr_grid(n, 'sym')                       # both fits use the same data
    p2s = [0.001, 0.002, 0.005, 0.01, 0.015] if not a.quick else [0.002, 0.005, 0.01]
    pro = 0.01; R = 100 if not a.quick else 30
    rng = np.random.default_rng(a.seed)
    cache = {}

    def probs(p2, lam=1.0):
        key = round(p2 * lam, 10)
        if key not in cache:
            cache[key] = np.array(pmap(_pA, [(L, M, phi, t, p2 * lam, p2 * lam / 10, pro) for t in th], a.nproc))
        return cache[key]

    rows = []; invariance = []
    for p2 in p2s:
        P = probs(p2)
        Pshift = np.array(pmap(_pA, [(L, M, phi, t + np.pi / 2, p2, p2 / 10, pro) for t in th[:5]], a.nproc))
        inv = dict(p2=p2, max_reflection_residual=float(np.abs(P - P[::-1]).max()),
                   max_period_residual=float(np.abs(Pshift - P[:5]).max()), P_min=float(P.min()))
        invariance.append(inv)
        print(f'p2={p2:.3f}: Pauli invariance  |P(t)-P(pi/2-t)| <= {inv["max_reflection_residual"]:.1e}, '
              f'|P(t+pi/2)-P(t)| <= {inv["max_period_residual"]:.1e}')
        for name, g in GRIDS.items():
            _, nu = fr.trig_fit(th, P, n, None, False, True, g['symmetric'])
            for floor in ['none', 'onaxis', 'min']:
                pairs, r, B = fr.reconstruct(th, P, n, None, floor, np.pi / 4, False, g['symmetric'])
                m = fr.match(pairs, zs)
                errs = []
                for _ in range(R):
                    pb, *_ = fr.sample_and_reconstruct(th, P, g['S'], n, rng, floor, np.pi / 4, False, g['symmetric'])
                    errs.append([(x['err_a'], x['err_theta']) for x in fr.match(pb, zs)])
                errs = np.array(errs)
                for i, z in enumerate(zs):
                    rows.append(dict(p2=p2, p_ro=pro, grid=name, floor=floor, B=B, nuisance_norm=nu, a0=z.real, theta0=z.imag,
                                     bias_a_inf=m[i]['err_a'], bias_theta_inf=m[i]['err_theta'],
                                     rms_a_S=float(np.sqrt((errs[:, i, 0] ** 2).mean())), rms_theta_S=float(np.sqrt((errs[:, i, 1] ** 2).mean())),
                                     S=g['S'], K=g['K']))
                near = [x for x in rows[-len(zs):] if abs(x['a0'] - 0.0213) < 1e-3][0]
                print(f'p2={p2:.3f} fit={name} floor={floor:6s} B={B:+.2e} sine-part {nu:.1e}: zero (+0.0213, 0.629) bias |a| {near["bias_a_inf"]:+.4f} '
                      f'theta {near["bias_theta_inf"]:+.4f}; rms at {g["K"]}x{g["S"]}: {near["rms_a_S"]:.4f} / {near["rms_theta_S"]:.4f}')
    # ZNE of positions (gate noise scaled by 1, 2, 3; readout fixed; floor 'onaxis')
    zne = []
    for p2 in p2s:
        for name, g in GRIDS.items():
            est = []
            for lam in (1, 2, 3):
                pairs, *_ = fr.reconstruct(th, probs(p2, lam), n, None, 'onaxis', np.pi / 4, False, g['symmetric'])
                est.append([(x['a_abs_hat'], x['theta_hat']) for x in fr.match(pairs, zs)])
            est = np.array(est)                         # [lam, zero, 2]
            lam = np.array([1, 2, 3])
            for i, z in enumerate(zs):
                ca = np.polyfit(lam, est[:, i, 0], 1); ct = np.polyfit(lam, est[:, i, 1], 1)
                zne.append(dict(p2=p2, grid=name, a0=z.real, theta0=z.imag, a_abs_lam=est[:, i, 0].tolist(), theta_lam=est[:, i, 1].tolist(),
                                a_abs_zne=float(ca[1]), theta_zne=float(ct[1]),
                                err_a_zne=float(ca[1] - abs(z.real)), err_theta_zne=circ_diff(ct[1], z.imag)))
            near = [x for x in zne[-len(zs):] if abs(x['a0'] - 0.0213) < 1e-3][0]
            print(f'ZNE p2={p2:.3f} fit={name}: zero (+0.0213, 0.629) |a|: x1 {near["a_abs_lam"][0]:.4f} -> ZNE {near["a_abs_zne"]:.4f} (exact 0.0213); '
                  f'theta err {near["err_theta_zne"]:+.4f}')
    save('v3_circuitnoise.json', dict(L=L, M=M, phi=phi, theta=th.tolist(), grids={k: {kk: (float(vv) if not isinstance(vv, bool) else vv) for kk, vv in v.items()} for k, v in GRIDS.items()},
                                      invariance=invariance, rows=rows, zne=zne), a.outdir)


# ----------------------------------------------------------------------------- V4
def v4(a):
    """Sign of a0 by two LCU points at the mirror candidates; ideal and noisy; error rate by simulation."""
    L, M, phi = 6, 3, 0.6
    n = fr.degree(L, M); zs = fr.exact_zeros(L, M, phi)
    th, g = fr_grid(n, a.grid)
    rng = np.random.default_rng(a.seed)
    p2s = [0.0, 0.002, 0.005, 0.01] if not a.quick else [0.0, 0.005]
    pro = 0.01
    rows = []
    for p2 in p2s:
        P = np.array(pmap(_pA, [(L, M, phi, t, p2, p2 / 10, pro if p2 else 0.0) for t in th], a.nproc))
        pairs, *_ = fr.reconstruct(th, P, n, None, 'onaxis' if p2 else 'none', np.pi / 4, False, g['symmetric'])
        for z in zs:
            if abs(z.real) < 1e-6 or abs(z.real) > 0.35:
                continue
            m = fr.match(pairs, [z])[0]
            pr = pairs[m['index']]
            tp, tm = fr.sign_test_points(pr)
            pp, pm = pmap(_pB, [(L, M, phi, tp[0], tp[1], p2, p2 / 10, pro if p2 else 0.0, True),
                                (L, M, phi, tm[0], tm[1], p2, p2 / 10, pro if p2 else 0.0, True)], a.nproc)
            S = fr.sign_test_shots(pp, pm, 1e-3)
            S_use = int(min(max(S, 1000), 2e6)) if np.isfinite(S) else int(2e6)
            truth_plus = z.real > 0
            dec = [(rng.binomial(S_use, pm) - rng.binomial(S_use, pp)) > 0 for _ in range(2000)]
            err_rate = float(np.mean([d != truth_plus for d in dec]))
            rows.append(dict(p2=p2, a0=z.real, theta0=z.imag, a_abs_hat=pr['a_abs'], theta_hat=pr['theta'], P_plus=pp, P_minus=pm,
                             shots_per_point_eps1e3=S, shots_used=S_use, simulated_error_rate=err_rate))
            print(f'p2={p2:.3f} zero ({z.real:+.4f},{z.imag:.4f}): test at |a|={pr["a_abs"]:.4f}: P(+)={pp:.2e} P(-)={pm:.2e}  '
                  f'shots/point {S:.0f}, simulated error rate {err_rate:.4f}')
    save('v4_signtest.json', dict(L=L, M=M, phi=phi, grid=a.grid, rows=rows), a.outdir)


# ----------------------------------------------------------------------------- V5
def v5(a):
    """Measurability window: rms error of (|a|, theta) vs |a0| over sizes at fixed shots per point."""
    sizes = [(4, 2), (4, 3), (6, 2), (6, 3), (6, 4), (8, 3), (8, 4), (10, 3), (10, 4)]
    if a.quick:
        sizes = [(4, 2), (6, 3), (8, 3), (8, 4)]
    phis = [0.4, 0.6]
    S = 50000; R = 100 if not a.quick else 30
    rng = np.random.default_rng(a.seed)
    rows = []
    for (L, M) in sizes:
        for phi in phis:
            n = fr.degree(L, M); zs = fr.exact_zeros(L, M, phi)
            th = fr.theta_grid(n); P = fr.return_probabilities_exact(L, M, phi, th)
            errs = []
            for _ in range(R):
                pairs, *_ = fr.sample_and_reconstruct(th, P, S, n, rng, symmetric=True)
                errs.append([(x['err_a'], x['err_theta']) for x in fr.match(pairs, zs)])
            errs = np.array(errs)
            for i, z in enumerate(zs):
                rows.append(dict(L=L, M=M, phi=phi, n=n, K=len(th), S=S, a0=z.real, theta0=z.imag,
                                 rms_a=float(np.sqrt((errs[:, i, 0] ** 2).mean())), rms_theta=float(np.sqrt((errs[:, i, 1] ** 2).mean())),
                                 lcu_cost_factor=float(np.exp(2 * L * M * abs(z.real)))))
            print(f'L={L:2d} M={M} phi={phi}: n={n}, zeros |a0| in [{min(abs(z.real) for z in zs):.3f},{max(abs(z.real) for z in zs):.3f}]; '
                  f'median rms |a| {np.median([r["rms_a"] for r in rows[-len(zs):]]):.4f}')
    save('v5_window.json', dict(rows=rows), a.outdir)
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(7.2, 3))
        for (L, M) in sizes:
            rr = [r for r in rows if r['L'] == L and r['M'] == M]
            ax[0].loglog([max(abs(r['a0']), 1e-4) for r in rr], [r['rms_a'] for r in rr], 'o', ms=3, label=f'{L}x{M}')
            ax[1].loglog([max(abs(r['a0']), 1e-4) for r in rr], [r['rms_theta'] for r in rr], 'o', ms=3)
        ax[0].set_xlabel(r'$|a_0|$'); ax[0].set_ylabel(r'rms error of $|a|$'); ax[1].set_xlabel(r'$|a_0|$'); ax[1].set_ylabel(r'rms error of $\theta$')
        ax[0].legend(fontsize=6, ncol=2); fig.suptitle(f'Fejer-Riesz reconstruction from return probabilities, {S} shots/point, K = 4n+4', fontsize=8)
        fig.tight_layout(); os.makedirs('figs', exist_ok=True); fig.savefig('figs/fr_window.png', dpi=150); print('-> figs/fr_window.png')
    except Exception as e:
        print('plot skipped:', e)


# ----------------------------------------------------------------------------- V6
def _lcu_grid(z0, h):
    offs = np.linspace(-h, h, 5)
    return [(z0.real + 0.3 * h + da, z0.imag - 0.2 * h + dt) for da in offs for dt in offs]


def v6(a):
    """Same circuit-level noise model for both routes, L=6 M=3 phi=0.6, targets: off-axis zero (+0.0213, 0.629)
    and on-axis zero (0, pi/4).  Fejer-Riesz: 40 theta points x 5e4 shots (+ sign test); LCU: 5x5 grid x 2e5."""
    from .analysis import fit_zero
    from .circuits import lcu_normalisation
    L, M, phi = 6, 3, 0.6
    n = fr.degree(L, M); zs = fr.exact_zeros(L, M, phi)
    targets = [z for z in zs if abs(z.real - 0.0213) < 1e-3 or (abs(z.real) < 1e-6)]
    p2s = [0.001, 0.002, 0.005, 0.01] if not a.quick else [0.002, 0.005]
    pro = 0.01; th, g = fr_grid(n, a.grid); S_fr = g['S']; S_lcu = 200000
    R_fr = 100 if not a.quick else 30; R_lcu = 30 if not a.quick else 5
    rng = np.random.default_rng(a.seed)
    rows = []
    for p2 in p2s:
        PA = np.array(pmap(_pA, [(L, M, phi, t, p2, p2 / 10, pro) for t in th], a.nproc))
        for z0 in targets:
            ef = []
            for _ in range(R_fr):
                pb, *_ = fr.sample_and_reconstruct(th, PA, S_fr, n, rng, 'onaxis', np.pi / 4, False, g['symmetric'])
                m = fr.match(pb, [z0])[0]
                ef.append(np.hypot(m['err_a'], m['err_theta']))
            for h in (0.03, 0.06):
                grid = _lcu_grid(z0, h)
                PB = np.array(pmap(_pB, [(L, M, phi, g[0], g[1], p2, p2 / 10, pro, True) for g in grid], a.nproc))
                w = [lcu_normalisation(L, M, g[0], 0.0) ** 2 for g in grid]
                el = []
                for _ in range(R_lcu):
                    k = rng.binomial(S_lcu, PB)
                    f = fit_zero(grid, k, np.full(25, S_lcu), weights=w)
                    # compare with the exact signed zero (LCU measures the sign directly)
                    el.append(abs(complex(f['z0'].real, f['z0'].imag) - z0))
                rows.append(dict(p2=p2, a0=z0.real, theta0=z0.imag, lcu_halfwidth=h,
                                 fr_median_err=float(np.median(ef)), fr_p90_err=float(np.percentile(ef, 90)), fr_shots=int(g['K'] * S_fr), fr_grid=a.grid,
                                 lcu_median_err=float(np.median(el)), lcu_p90_err=float(np.percentile(el, 90)), lcu_shots=int(25 * S_lcu)))
                print(f'p2={p2:.3f} zero ({z0.real:+.4f},{z0.imag:.4f}) | FR (|a|,theta), 2e6 shots: median err {np.median(ef):.4f} | '
                      f'LCU 5x5 hw={h}, 5e6 shots: median err {np.median(el):.4f}')
    save('v6_fr_vs_lcu.json', dict(L=L, M=M, phi=phi, rows=rows, note='FR error is in (|a|, theta); sign from V4'), a.outdir)


# ----------------------------------------------------------------------------- V7
def v7(a):
    """Reset redundancy: joint probability with vs without reset after each ancilla measurement."""
    rows = []
    for (L, M) in [(4, 2), (6, 3)]:
        for aa in (0.1, -0.1):
            for (p2, pro) in [(0.0, 0.0), (0.0, 0.01), (0.0, 0.03), (0.005, 0.01)]:
                th, phi = 0.8, 0.6
                with_r, without_r = pmap(_pB, [(L, M, phi, aa, th, p2, p2 / 10, pro, True), (L, M, phi, aa, th, p2, p2 / 10, pro, False)], a.nproc)
                rows.append(dict(L=L, M=M, a=aa, p2=p2, p_ro=pro, P_with_reset=with_r, P_without_reset=without_r,
                                 rel_diff=abs(with_r - without_r) / with_r))
                print(f'L={L} M={M} a={aa:+.1f} p2={p2} p_ro={pro}: with reset {with_r:.6e}  without {without_r:.6e}  rel. diff {abs(with_r-without_r)/with_r:.2e}')
    save('v7_reset.json', dict(rows=rows), a.outdir)


# ----------------------------------------------------------------------------- V8
def _v8_one(args):
    grid, PB, S, w, nboot, seed, z0 = args
    from .analysis import bootstrap_zero
    rng = np.random.default_rng(seed)
    k = rng.binomial(S, PB)
    fit, zb = bootstrap_zero(grid, k, np.full(len(grid), S), nboot=nboot, rng=rng, weights=w)
    lo_a, hi_a = np.percentile(zb.real, [2.5, 97.5]); lo_t, hi_t = np.percentile(zb.imag, [2.5, 97.5])
    return dict(a_hat=fit['z0'].real, t_hat=fit['z0'].imag, in_a=bool(lo_a <= z0.real <= hi_a), in_t=bool(lo_t <= z0.imag <= hi_t),
                w_a=hi_a - lo_a, w_t=hi_t - lo_t)


def v8(a):
    """Coverage of the LCU parametric-bootstrap 95 % intervals under circuit-level noise."""
    from .circuits import lcu_normalisation
    L, M, phi = 6, 3, 0.6
    zs = fr.exact_zeros(L, M, phi)
    z_on = [z for z in zs if abs(z.real) < 1e-6][0]
    z_off = [z for z in zs if abs(z.real - 0.0213) < 1e-3][0]          # the planned WP3b target
    cases = ([(z_off, 0.0, 0.0, 0.03), (z_off, 0.001, 0.01, 0.03), (z_off, 0.002, 0.01, 0.03), (z_on, 0.002, 0.01, 0.06)]
             if not a.quick else [(z_off, 0.001, 0.01, 0.03)])
    R = 200 if not a.quick else 8; nboot = 100 if not a.quick else 20; S = 200000
    rows = []
    for (z0, p2, pro, h) in cases:
        grid = _lcu_grid(z0, h)
        PB = np.array(pmap(_pB, [(L, M, phi, g[0], g[1], p2, p2 / 10, pro, True) for g in grid], a.nproc))
        w = [lcu_normalisation(L, M, g[0], 0.0) ** 2 for g in grid]
        res = pmap(_v8_one, [(grid, PB, S, w, nboot, a.seed + i, z0) for i in range(R)], a.nproc)
        cov_a = np.mean([r['in_a'] for r in res]); cov_t = np.mean([r['in_t'] for r in res])
        cov_joint = np.mean([r['in_a'] and r['in_t'] for r in res])
        bias = (np.mean([r['a_hat'] for r in res]) - z0.real, np.mean([r['t_hat'] for r in res]) - z0.imag)
        rows.append(dict(target_a=z0.real, target_theta=z0.imag, p2=p2, p_ro=pro, halfwidth=h, R=R, nboot=nboot, coverage_a=cov_a, coverage_theta=cov_t, coverage_joint=cov_joint,
                         mean_bias_a=bias[0], mean_bias_theta=bias[1], median_width_a=float(np.median([r['w_a'] for r in res])),
                         median_width_theta=float(np.median([r['w_t'] for r in res]))))
        print(f'target ({z0.real:+.4f},{z0.imag:.4f}) p2={p2} p_ro={pro} hw={h}: coverage a {cov_a:.2f}, theta {cov_t:.2f}, joint {cov_joint:.2f} (nominal 0.95 / ~0.90); '
              f'mean bias ({bias[0]:+.4f}, {bias[1]:+.4f})')
    save('v8_lcu_coverage.json', dict(L=L, M=M, phi=phi, rows=rows), a.outdir)


# ----------------------------------------------------------------------------- report
def report(a):
    import glob
    lines = ['# Verification report (isingzeros rev5)', '', f'generated {time.strftime("%Y-%m-%d %H:%M")}', '']
    for f in sorted(glob.glob(os.path.join(a.outdir, 'v*.json'))):
        d = json.load(open(f))
        lines.append(f'## {os.path.basename(f)}'); lines.append('')
        rows = d.get('rows', [])
        if rows:
            keys = [k for k in rows[0].keys() if not isinstance(rows[0][k], (list, dict))]
            lines.append('| ' + ' | '.join(keys) + ' |'); lines.append('|' + '---|' * len(keys))
            for r in rows[:400]:
                lines.append('| ' + ' | '.join(f'{r[k]:.4g}' if isinstance(r[k], float) else str(r[k]) for k in keys) + ' |')
        for k in ('coverage', 'invariance', 'zne'):
            if k in d:
                lines.append(''); lines.append(f'{k}: `{json.dumps(d[k])[:3000]}`')
        lines.append('')
    open(os.path.join(a.outdir, 'REPORT.md'), 'w').write('\n'.join(lines))
    print('->', os.path.join(a.outdir, 'REPORT.md'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('test', choices=['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'report'])
    ap.add_argument('--quick', action='store_true', help='reduced sizes / repetitions (minutes on 2 cores)')
    ap.add_argument('--nproc', type=int, default=1)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--outdir', default=OUT)
    ap.add_argument('--grid', default='sym', choices=['sym', 'gen'], help="fit design for V4/V6 (see GRIDS)")
    a = ap.parse_args()
    t0 = time.time()
    globals()[a.test](a)
    print(f'[{a.test}] done in {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
