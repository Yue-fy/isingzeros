"""
Resource budget for the hardware campaigns, computed from EXACT return probabilities at the planned
parameter points (transfer matrix), not from a "typical P".

    python -m isingzeros.budget --out results/budget.csv [--eps 0.2] [--shot_ms 1.0]

For every planned point the shots needed for a relative standard error eps on P are
    S(P) = (1 - P) / (P eps^2)            (binomial),
capped at --max_shots.  For the LCU protocol the joint event is "all ancillas 0 AND return", with
probability  P_joint = P_succ * P(return|succ) = N^2 |<+|U(K)|+>|^2,  and the shots needed are
S(P_joint); the expected number of joint events is reported explicitly.

Three accounting units are reported, each with its stated assumption:
    device-seconds        = shots * shot_ms / 1000              (shot_ms: wall time per shot, assumed)
    qubit-seconds (used)  = device-seconds * n_qubits_in_circuit
    qubit-seconds (whole) = device-seconds * 54                   (if the site bills the whole machine)
Piast-Q is billed per job (200 shots/job, 60-145 s/job including ion loading): job counts are given.
"""
from __future__ import annotations
import argparse, csv, os
import numpy as np
from .classical import RowTransfer, primal_coupling
from .circuits import kicked_ising_unitary, kicked_ising_lcu, kicked_ising_hadamard_test, gate_counts, lcu_normalisation


def resolvable_fraction(P, n, cz, p2, p_ro):
    """Fraction of planned points whose attenuated signal exceeds 3x the depolarised noise floor:
    signal = P * F * (1-p_ro)^n,  floor = (1-F) / 2^n,  F = (1-p2)^cz  (pre-mitigation, independent errors)."""
    F = (1 - p2) ** cz
    sig = np.asarray(P) * F * (1 - p_ro) ** n
    floor = (1 - F) / 2 ** n
    return float(np.mean(sig > 3 * floor))


def load_routed():
    """Routed CZ counts on the IQM 54-qubit lattice (results/routing.json), keyed by (L, M, W)."""
    try:
        import json
        d = json.load(open('results/routing.json'))
        out = {}
        for r in d['rows']:
            if 'cz_routed' in r and r['case'].startswith('A '):
                out[(r['L'], r['M'], r['W'])] = r['cz_routed']
            if 'cz_routed' in r and r['case'].startswith('B LCU') and 'MCM' in r['case']:
                out[('lcu', r['L'], r['M'])] = r['cz_routed']
        return out
    except Exception:
        return {}


NPROC = 1        # process pool for the exact probabilities (set by --nproc)


def _P_exact(args):
    L, M, W, phi, theta3, th = args
    return abs(RowTransfer(L, primal_coupling(1j * phi), 1j * th, W=W, K3=1j * theta3).strip_amplitude(M)) ** 2


def _P_many(L, M, W, phi, theta3, thetas):
    args = [(L, M, W, phi, theta3, float(th)) for th in thetas]
    if NPROC > 1 and len(args) > 1:
        from multiprocessing import Pool
        with Pool(min(NPROC, len(args))) as pool:
            return np.array(pool.map(_P_exact, args))
    return np.array([_P_exact(x) for x in args])


def theta_points(L, M, W, phi, theta3, n_coarse=13, n_fine=8):
    """Coarse grid on [0.05, pi/2 - 0.05] plus fine points around the two deepest minima of P (exact)."""
    coarse = np.linspace(0.05, np.pi / 2 - 0.05, n_coarse)
    P = _P_many(L, M, W, phi, theta3, coarse)
    idx = np.argsort(P)[:2]
    fine = np.concatenate([np.linspace(coarse[i] - 0.08, coarse[i] + 0.08, n_fine // 2 + 1)[1:-1] for i in idx])
    pts = np.unique(np.concatenate([coarse, fine]))
    return pts, _P_many(L, M, W, phi, theta3, pts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/budget.csv')
    ap.add_argument('--eps', type=float, default=0.2, help='target relative std error on P at each point')
    ap.add_argument('--max_shots', type=int, default=200000)
    ap.add_argument('--min_shots', type=int, default=2000)
    ap.add_argument('--shot_ms', type=float, default=1.0, help='assumed wall time per shot on superconducting hardware')
    ap.add_argument('--p_ro', type=float, default=0.01)
    ap.add_argument('--nproc', type=int, default=1, help='process pool for the exact probabilities (~1 GB per worker at n = 24)')
    a = ap.parse_args()
    global NPROC
    NPROC = a.nproc
    rows = []
    routed_cz = load_routed()

    # ---------------- Campaign A: chains on Euro-Q-Exa (protocol A), two phi values, zero-focused theta grid
    planA = [(8, 3, 1, 0.0), (8, 4, 1, 0.0), (12, 4, 1, 0.0), (16, 4, 1, 0.0), (16, 6, 1, 0.0), (24, 4, 1, 0.0)]
    # ---------------- Campaign A2: aspect-ratio control (small L and M comparable)
    planA2 = [(4, 4, 1, 0.0), (6, 6, 1, 0.0), (8, 8, 1, 0.0)]
    # ---------------- Campaign B: finite-thickness slabs (W = 2, 3) on Euro-Q-Exa
    planB = [(8, 3, 2, 0.3), (12, 4, 2, 0.3), (8, 3, 3, 0.3), (16, 3, 2, 0.3)]
    phis = [0.4, 0.6]
    lam_ref = {}      # rate functions per site of the largest exactly computed slab, for extrapolation
    for name, plan in [('A_chain', planA), ('A2_aspect', planA2), ('B_slab', planB)]:
        for (L, M, W, th3) in plan:
            g = gate_counts(kicked_ising_unitary(L, M, 0.4, 0.4, W=W, theta3=th3))
            for phi in phis:
                n = L * W
                if n <= 24:
                    nc, nf = (13, 8) if n <= 16 else (9, 4)
                    pts, P = theta_points(L, M, W, phi, th3, n_coarse=nc, n_fine=nf)
                    if W > 1:
                        lam_ref[(phi, W)] = (-np.log(P) / (2 * n), pts)
                else:   # beyond exact reach here: extrapolate P = exp(-2 n lambda) from the largest exact slab
                    lam, pts = lam_ref[(phi, W)]
                    P = np.exp(-2 * n * lam)
                shots = np.clip((1 - P) / (P * a.eps ** 2), a.min_shots, a.max_shots).astype(int)
                tot = int(shots.sum())
                dev_s = tot * a.shot_ms / 1000
                cz_r = routed_cz.get((L, M, W), g['two_qubit'])
                res = {p2: resolvable_fraction(P, n, cz_r, p2, a.p_ro) for p2 in (0.005, 0.015)}
                rows.append(dict(campaign=name, system='Euro-Q-Exa', protocol='A_return_prob', L=L, M=M, W=W, theta3=th3, phi=phi,
                                 qubits=L * W, cz=g['two_qubit'], cz_routed=cz_r, depth=g['depth'], points=len(pts),
                                 P_min=float(P.min()), P_median=float(np.median(P)), shots_total=tot,
                                 frac_resolvable_p2_0p5=res[0.005], frac_resolvable_p2_1p5=res[0.015],
                                 device_s=dev_s, qubit_s_used=dev_s * L * W, qubit_s_whole54=dev_s * 54, jobs200=0,
                                 note=f'{a.eps:.0%} rel. error per point; shots capped at {a.max_shots}' + ('; P extrapolated from smaller slab' if L * W > 24 else '')))

    # ---------------- Campaign C (revision 4): off-axis zeros, L = 6, M = 3, phi = 0.6
    #   C_FR    (WP3a, no MCM): Fejer-Riesz scan of protocol A on one period [0, pi/2): 40 theta points x 5e4 shots
    #           (grid shifted by half a step); control L = 4, M = 2: 40 points x 2.5e4
    #   C_sign  (WP3a): two-point sign test (LCU, MCM or deferred) for each near-axis off-axis zero, 5e4 shots/point
    #   C_zero2D(WP3b, needs MCM + floor go/no-go): ONE local 5x5 LCU grid around (+0.0213, 0.629), 2e5 shots/point;
    #           the other targets are conditional (not in the base plan).  L = 8 off-axis localisation is removed.
    from .analysis import joint_probability
    from . import fejer
    for (L, M, K, S) in [(6, 3, 40, 50000), (4, 2, 40, 25000)]:
        phi = 0.6
        g = gate_counts(kicked_ising_unitary(L, M, 0.6, phi))
        th = fejer.theta_grid(fejer.degree(L, M), K)
        P = fejer.return_probabilities_exact(L, M, phi, th)
        tot = K * S; dev_s = tot * a.shot_ms / 1000
        cz_r = routed_cz.get((L, M, 1), g['two_qubit'])
        res = {p2: resolvable_fraction(P, L, cz_r, p2, a.p_ro) for p2 in (0.005, 0.015)}
        rows.append(dict(campaign='C_FR', system='Euro-Q-Exa', protocol='A_return_prob_FR', L=L, M=M, W=1, theta3=0.0, phi=phi,
                         qubits=L, cz=g['two_qubit'], cz_routed=cz_r, depth=g['depth'], points=K,
                         P_min=float(P.min()), P_median=float(np.median(P)), shots_total=tot,
                         frac_resolvable_p2_0p5=res[0.005], frac_resolvable_p2_1p5=res[0.015],
                         device_s=dev_s, qubit_s_used=dev_s * L, qubit_s_whole54=dev_s * 54, jobs200=0,
                         note=f'WP3a Fejer-Riesz scan, {K} theta points on [0, pi/2), {S} shots each; no MCM'))
    L, M, phi = 6, 3, 0.6
    near = [z for z in fejer.exact_zeros(L, M, phi) if 0.005 < abs(z.real) < 0.1]
    g = gate_counts(kicked_ising_lcu(L, M, 0.8, phi, 0.1, 0.0))
    cz_r = routed_cz.get(('lcu', L, M), g['two_qubit'])
    for z0 in near:
        Pp = joint_probability(L, M, phi, abs(z0.real), z0.imag); Pm = joint_probability(L, M, phi, -abs(z0.real), z0.imag)
        tot = 2 * 50000; dev_s = tot * a.shot_ms / 1000
        rows.append(dict(campaign='C_sign', system='Euro-Q-Exa', protocol='B_lcu_sign_test', L=L, M=M, W=1, theta3=0.0, phi=phi,
                         qubits=L + 1, cz=g['two_qubit'], cz_routed=cz_r, depth=g['depth'], points=2,
                         P_min=float(min(Pp, Pm)), P_median=float(max(Pp, Pm)), shots_total=tot,
                         frac_resolvable_p2_0p5=float('nan'), frac_resolvable_p2_1p5=float('nan'),
                         device_s=dev_s, qubit_s_used=dev_s * (L + 1), qubit_s_whole54=dev_s * 54, jobs200=0,
                         note=f'WP3a sign test at (+-{abs(z0.real):.4f}, {z0.imag:.4f}); ideal P_joint {Pp:.2e} / {Pm:.2e}; 5e4 shots/point'))
    zeros2d = sorted([z for z in fejer.exact_zeros(L, M, phi) if abs(z.real) <= 0.2], key=lambda z: (-int(z.real > 1e-6), abs(z.real)))
    for j, z0 in enumerate(zeros2d[:3]):
        h = 0.03; offs = np.linspace(-h, h, 5)      # grid deliberately not centred on the zero (as in analysis.demo)
        pts = [(z0.real + 0.3 * h + da, z0.imag - 0.2 * h + dt) for da in offs for dt in offs]
        Pj = np.array([joint_probability(L, M, phi, p[0], p[1]) for p in pts])
        tot = 25 * a.max_shots; dev_s = tot * a.shot_ms / 1000; ev = Pj * a.max_shots
        rows.append(dict(campaign='C_zero2D' if j == 0 else 'C_zero2D_conditional', system='Euro-Q-Exa', protocol='B_lcu_mcm', L=L, M=M, W=1, theta3=0.0, phi=phi,
                         qubits=L + 1, cz=g['two_qubit'], cz_routed=cz_r, depth=g['depth'], points=25,
                         P_min=float(Pj.min()), P_median=float(np.median(Pj)), shots_total=tot,
                         frac_resolvable_p2_0p5=float('nan'), frac_resolvable_p2_1p5=float('nan'),
                         device_s=dev_s, qubit_s_used=dev_s * (L + 1), qubit_s_whole54=dev_s * 54, jobs200=0,
                         note=f'WP3b target zero a0={z0.real:.4f}, theta0={z0.imag:.4f}; 5x5 grid hw=0.03 offset (0.3h,-0.2h); expected joint events per point: min {ev.min():.0f}, median {np.median(ev):.0f}, max {ev.max():.0f} (ideal)'))

    # ---------------- Piast-Q: protocol A (all-to-all, RXX native) and Hadamard test; no mid-circuit measurement
    for (L, M) in [(8, 3), (12, 3)]:
        pts, P = theta_points(L, M, 1, 0.6, 0.0, n_coarse=11, n_fine=4)
        shots = np.clip((1 - P) / (P * a.eps ** 2), a.min_shots, 20000).astype(int)
        tot = int(shots.sum()); jobs = int(np.ceil(tot / 200))
        g = gate_counts(kicked_ising_unitary(L, M, 0.4, 0.6), basis=('rz', 'r', 'rxx', 'measure'))
        rows.append(dict(campaign='P_chain', system='Piast-Q', protocol='A_return_prob', L=L, M=M, W=1, theta3=0.0, phi=0.6,
                         qubits=L, cz=dict(g['ops']).get('rxx', 0), depth=g['depth'], points=len(pts),
                         P_min=float(P.min()), P_median=float(np.median(P)), shots_total=tot, device_s=jobs * 100.0,
                         cz_routed=dict(g['ops']).get('rxx', 0), frac_resolvable_p2_0p5=resolvable_fraction(P, L, dict(g['ops']).get('rxx', 0), 0.005, a.p_ro), frac_resolvable_p2_1p5=resolvable_fraction(P, L, dict(g['ops']).get('rxx', 0), 0.015, a.p_ro),
                         qubit_s_used=0, qubit_s_whole54=0, jobs200=jobs, note='RXX count in cz column; 100 s/job assumed (60-145 s)'))
    for (L, M) in [(4, 2)]:
        g = gate_counts(kicked_ising_hadamard_test(L, M, 0.4, 0.6), basis=('rz', 'r', 'rxx', 'measure'))
        pts = 9; shots = 4000 * 2   # Re and Im
        tot = pts * shots; jobs = int(np.ceil(tot / 200))
        rows.append(dict(campaign='P_hadamard', system='Piast-Q', protocol='C_hadamard', L=L, M=M, W=1, theta3=0.0, phi=0.6,
                         qubits=L + 1, cz=dict(g['ops']).get('rxx', 0), depth=g['depth'], points=pts,
                         P_min=float('nan'), P_median=float('nan'), shots_total=tot, device_s=jobs * 100.0,
                         cz_routed=dict(g['ops']).get('rxx', 0), frac_resolvable_p2_0p5=float('nan'), frac_resolvable_p2_1p5=float('nan'),
                         qubit_s_used=0, qubit_s_whole54=0, jobs200=jobs, note='optional (P2); complex amplitude, +-0.016 abs. error per quadrature'))

    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    # summary
    print(f'{"campaign":10s} {"sys":10s} {"L":>3s} {"M":>2s} {"W":>2s} {"phi":>4s} {"nq":>3s} {"2q":>4s} {"2qR":>4s} {"pts":>3s} {"Pmin":>9s} {"shots":>10s} {"dev_h":>7s} {"qb_h_used":>10s} {"qb_h_54":>8s} {"jobs":>6s} {"res.5%":>6s} {"res1.5%":>7s}')
    tot = {}
    for r in rows:
        print(f'{r["campaign"]:10s} {r["system"]:10s} {r["L"]:3d} {r["M"]:2d} {r["W"]:2d} {r["phi"]:4.1f} {r["qubits"]:3d} {r["cz"]:4d} {r.get("cz_routed", 0):4d} {r["points"]:3d} {r["P_min"]:9.2e} {r["shots_total"]:10d} {r["device_s"]/3600:7.2f} {r["qubit_s_used"]/3600:10.1f} {r["qubit_s_whole54"]/3600:8.1f} {r["jobs200"]:6d} {r.get("frac_resolvable_p2_0p5", float("nan")):6.2f} {r.get("frac_resolvable_p2_1p5", float("nan")):7.2f}')
        cond = 'conditional' in r['campaign'] or (r['campaign'] == 'B_slab' and r['L'] * r['W'] > 24)
        t = tot.setdefault(r['system'] + (' CONDITIONAL' if cond else ' base'), dict(shots=0, dev=0.0, qu=0.0, q54=0.0, jobs=0))
        t['shots'] += r['shots_total']; t['dev'] += r['device_s']; t['qu'] += r['qubit_s_used']; t['q54'] += r['qubit_s_whole54']; t['jobs'] += r['jobs200']
    for s, t in tot.items():
        print(f'TOTAL {s}: shots={t["shots"]:.3e}  device-hours={t["dev"]/3600:.1f}  qubit-hours(used)={t["qu"]/3600:.0f}  qubit-hours(whole 54)={t["q54"]/3600:.0f}  jobs={t["jobs"]}')
    print(f'-> {a.out}')


if __name__ == '__main__':
    main()
