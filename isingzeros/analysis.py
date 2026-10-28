"""
Localisation of a finite-size zero in the complex K2 plane from counted events on a local 2D grid.

Model.  Near a simple zero z0 of the (holomorphic) strip amplitude A(z), z = K2 = a + i theta,
    A(z) = c (z - z0) + O((z - z0)^2),
so the joint probability measured on hardware is
    P(a, theta) = G |z - z0|^2 + B,
with G = N^2 |c|^2 (N the LCU normalisation) and B >= 0 a noise floor (depolarised background,
readout leakage, post-selection bias).  Four real parameters (Re z0, Im z0, G, B) are fitted to the
raw counts k_i ~ Binomial(S_i, P_i) by maximum likelihood (Poisson approximation is fine for P << 1);
confidence regions by parametric bootstrap.  Zero-count points enter through the likelihood, not
through -log P, and a point that is "compatible with zero" is reported with its 95 % upper limit
3/S (zero counts) rather than as a located zero.

    python -m isingzeros.analysis demo --L 6 --M 3 --phi 0.6 --shots 200000 --floor 2e-4 --out results/zero_localisation_demo.json
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
from scipy.optimize import minimize
from .classical import RowTransfer, primal_coupling, find_zeros_on_grid
from .circuits import lcu_normalisation


def exact_zeros_K2plane(L, M, phi, W=1, theta3=0.0, a_max=0.3, nre=121, nim=61):
    """Zeros of the strip amplitude in the complex K2 = a + i theta plane (|a| <= a_max, 0 < theta < pi/2)."""
    K1 = primal_coupling(1j * phi)
    f = lambda z: RowTransfer(L, K1, z, W=W, K3=1j * theta3).strip_amplitude(M)      # z = a + i theta
    zs = find_zeros_on_grid(lambda w: f(1j * w), (0.0, np.pi / 2), (-a_max, a_max), nre=nre, nim=nim)
    # w = theta_r + i theta_i  <->  K2 = i w = -theta_i + i theta_r  => a = -Im w, theta = Re w
    return [complex(-w.imag, w.real) for w in zs]


def joint_probability(L, M, phi, a, theta, W=1, theta3=0.0):
    """P_joint = N^2 |<+|U(K)|+>|^2 at K2 = a + i theta (LCU with real part a on all ZZ steps)."""
    K1 = primal_coupling(1j * phi)
    amp = RowTransfer(L, K1, a + 1j * theta, W=W, K3=1j * theta3).strip_amplitude(M)
    return lcu_normalisation(L, M, a, 0.0, W) ** 2 * abs(amp) ** 2


def noisy_joint_probability_dm(qc, p2=5e-3, p1=5e-4, p_ro=1e-2):
    """Exact joint probability P(all ancilla outcomes 0 AND return) of an LCU circuit under the generic
    circuit-level noise model of `simulate.generic_noise` (depolarising p2 after every two-qubit gate, p1
    after every physical one-qubit gate (rz is virtual and error-free), symmetric readout error p_ro), computed with an (unnormalised) density
    matrix instead of shot-by-shot trajectories: mid-circuit post-selection is the non-trace-preserving
    Kraus map {sqrt(1-p_ro)|0><0|, sqrt(p_ro)|1><1|} (outcome read as 0; any `reset` in the circuit is applied
    as its own channel), and the final X-basis
    readout error is applied to the diagonal.  Exact for n <= ~10 qubits; seconds per point."""
    from qiskit.quantum_info import DensityMatrix, Kraus, Operator
    from qiskit_aer.noise import depolarizing_error
    from .circuits import compile_fixed
    t = compile_fixed(qc)          # angle-independent gate sequence (see circuits.compile_fixed)
    n = t.num_qubits
    dep = {1: Kraus(depolarizing_error(p1, 1).to_quantumchannel()), 2: Kraus(depolarizing_error(p2, 2).to_quantumchannel())}
    # mid-circuit measurement post-selected on READING 0 (no implicit reset): true outcome 0 read correctly
    # (prob 1-p_ro) leaves |0>; true outcome 1 misread as 0 (prob p_ro) leaves |1>.  An explicit `reset`
    # instruction in the circuit is applied separately below.
    post = Kraus([np.sqrt(1 - p_ro) * np.array([[1, 0], [0, 0]]), np.sqrt(p_ro) * np.array([[0, 0], [0, 1]])])
    reset = Kraus([np.array([[1, 0], [0, 0]]), np.array([[0, 1], [0, 0]])])
    # which measurements are final (nothing but measurements act on that qubit afterwards)?
    data = list(t.data)
    last_nonmeasure = {}
    for i, inst in enumerate(data):
        if inst.operation.name not in ('measure', 'barrier'):
            for q in inst.qubits:
                last_nonmeasure[t.find_bit(q).index] = i
    rho = DensityMatrix.from_label('0' * n)
    final = []
    for i, inst in enumerate(data):
        name = inst.operation.name; qs = [t.find_bit(q).index for q in inst.qubits]
        if name == 'barrier':
            continue
        if name == 'measure':
            if last_nonmeasure.get(qs[0], -1) > i:
                rho = rho.evolve(post, qs)          # mid-circuit post-selection on 0 (with readout error)
            else:
                final.append(qs[0])
        elif name == 'reset':
            rho = rho.evolve(reset, qs)
        else:
            rho = rho.evolve(Operator(inst.operation), qs)
            if (p1 > 0 or p2 > 0) and name != 'rz':     # rz is a virtual frame change on IQM/IBM hardware: no error
                rho = rho.evolve(dep[len(qs)], qs)
    diag = np.real(np.diag(rho.data))
    idx = np.arange(2 ** n)
    weight = np.ones(2 ** n)
    for q in final:
        bit = (idx >> q) & 1
        weight *= np.where(bit == 0, 1 - p_ro, p_ro)
    return float(np.sum(diag * weight))


def fit_zero(points, counts, shots, z_init=None, weights=None):
    """points: list of (a, theta); counts k_i; shots S_i; weights w_i = known deterministic factor
    multiplying |A|^2 at each point (the LCU normalisation N(a_i)^2, which has a kink at a = 0 and
    must NOT be absorbed into the smooth model).  Returns dict with z0, G, B, d, nll."""
    pts = np.array(points); k = np.array(counts, float); S = np.array(shots, float)
    z = pts[:, 0] + 1j * pts[:, 1]
    w = np.ones(len(z)) if weights is None else np.array(weights, float)
    if z_init is None:
        z_init = z[np.argmin(k / S)]

    def nll(x):
        # P = G |z - z0|^2 |1 + d (z - z0)|^2 + B : simple zero with the next Taylor term (complex d),
        # which absorbs the curvature caused by neighbouring zeros within the fitting window.
        z0 = x[0] + 1j * x[1]; G = np.exp(x[2]); B = np.exp(x[3]); d = x[4] + 1j * x[5]
        P = np.clip(w * G * np.abs(z - z0) ** 2 * np.abs(1 + d * (z - z0)) ** 2 + B, 1e-15, 1 - 1e-15)
        return -np.sum(k * np.log(P) + (S - k) * np.log(1 - P))

    p0 = k / S
    G0 = max(np.median((p0 / w)[p0 > 0]) / max(np.median(np.abs(z - z_init) ** 2), 1e-6), 1e-9)
    x0 = [z_init.real, z_init.imag, np.log(G0), np.log(max(p0.min(), 1e-7)), 0.0, 0.0]
    best = None
    for dz in [(0, 0), (0.01, 0), (-0.01, 0), (0, 0.01), (0, -0.01)]:
        r = minimize(nll, [x0[0] + dz[0], x0[1] + dz[1], x0[2], x0[3], 0.0, 0.0], method='Nelder-Mead',
                     options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 40000, 'maxfev': 40000})
        if best is None or r.fun < best.fun:
            best = r
    x = best.x
    return dict(z0=complex(x[0], x[1]), G=float(np.exp(x[2])), B=float(np.exp(x[3])), d=complex(x[4], x[5]), nll=float(best.fun))


def bootstrap_zero(points, counts, shots, nboot=200, rng=None, weights=None):
    """Parametric bootstrap: resample counts from the fitted model, refit, return z0 samples."""
    rng = np.random.default_rng(rng)
    fit = fit_zero(points, counts, shots, weights=weights)
    pts = np.array(points); z = pts[:, 0] + 1j * pts[:, 1]; S = np.array(shots)
    w = np.ones(len(z)) if weights is None else np.array(weights, float)
    P = np.clip(w * fit['G'] * np.abs(z - fit['z0']) ** 2 * np.abs(1 + fit['d'] * (z - fit['z0'])) ** 2 + fit['B'], 0, 1)
    zs = []
    for _ in range(nboot):
        kb = rng.binomial(S.astype(int), P)
        zs.append(fit_zero(points, kb, shots, z_init=fit['z0'], weights=weights)['z0'])
    zs = np.array(zs)
    return fit, zs


def demo(a):
    zeros = exact_zeros_K2plane(a.L, a.M, a.phi)
    zeros = [z for z in zeros if abs(z.real) <= 0.2 and 0.2 < z.imag < 1.4]
    if not zeros:
        raise SystemExit('no zero with |a| <= 0.2 found; increase a_max')
    z0 = min(zeros, key=lambda z: abs(z.real))          # the zero closest to the imaginary axis
    print(f'exact zeros with |a|<=0.2: {[(round(z.real,4), round(z.imag,4)) for z in zeros]}')
    print(f'target zero z0 = a0 + i theta0 = {z0.real:.4f} + i {z0.imag:.4f}')
    # local 5x5 grid of half-width h around z0 (grid intentionally not centred exactly on z0)
    h = a.halfwidth
    offs = np.linspace(-h, h, 5)
    pts = [(z0.real + 0.3 * h + da, z0.imag - 0.2 * h + dt) for da in offs for dt in offs]
    Pex = np.array([joint_probability(a.L, a.M, a.phi, p[0], p[1]) for p in pts])
    rng = np.random.default_rng(a.seed)
    shots = np.full(len(pts), a.shots)
    if a.dm_p2 > 0:
        # circuit-level noise (depolarising p2 after CZ, p2/10 after 1q gates, readout p_ro) evaluated exactly
        # with the density-matrix method, then binomial counts
        from .circuits import kicked_ising_lcu
        Pn = np.array([noisy_joint_probability_dm(kicked_ising_lcu(a.L, a.M, th, a.phi, aa, 0.0), a.dm_p2, a.dm_p2 / 10, a.aer_pro)
                       for (aa, th) in pts])
        counts = rng.binomial(shots, Pn)
        print(f'noisy/ideal joint probability on the grid: min {np.min(Pn/Pex):.3f}, max {np.max(Pn/Pex):.3f}; noisy minimum {Pn.min():.2e} (ideal {Pex.min():.2e})')
    elif a.aer_p2 > 0:
        # full circuit-level noisy simulation of the LCU circuit (depolarising p2 on CZ, readout error),
        # joint counts = shots with all ancilla outcomes 0 AND return  (slow: shot-by-shot trajectories)
        from .simulate import run_counts, generic_noise
        from .circuits import kicked_ising_lcu, lcu_postselect
        noise = generic_noise(p2=a.aer_p2, p1=a.aer_p2 / 10, p_ro=a.aer_pro)
        counts = []
        for i, (aa, th) in enumerate(pts):
            qc = kicked_ising_lcu(a.L, a.M, th, a.phi, aa, 0.0)
            c, _ = run_counts(qc, a.shots, noise, a.seed + i)
            p_succ, p_ret = lcu_postselect(c, a.L)
            counts.append(int(round(p_succ * p_ret * a.shots)))
            print(f'  point {i+1}/{len(pts)} a={aa:.4f} theta={th:.4f}  joint counts {counts[-1]}  (ideal {Pex[i]*a.shots:.0f})')
        counts = np.array(counts)
    else:
        Pobs = np.clip(Pex * a.signal + a.floor, 0, 1)      # signal attenuation + noise floor (hardware model)
        counts = rng.binomial(shots, Pobs)
    weights = [lcu_normalisation(a.L, a.M, p[0], 0.0) ** 2 for p in pts]          # known N(a)^2 per point
    fit, zb = bootstrap_zero(pts, counts, shots, nboot=a.nboot, rng=rng, weights=weights)
    err = abs(fit['z0'] - z0)
    ci_a = np.percentile(zb.real, [2.5, 97.5]); ci_t = np.percentile(zb.imag, [2.5, 97.5])
    print(f'fit: z0 = {fit["z0"].real:.4f} + i {fit["z0"].imag:.4f}   (error {err:.4f});  floor B = {fit["B"]:.2e} (true {a.floor:.1e})')
    print(f'95% CI: a in [{ci_a[0]:.4f}, {ci_a[1]:.4f}],  theta in [{ci_t[0]:.4f}, {ci_t[1]:.4f}]')
    print(f'counts on the grid: min {counts.min()}, median {int(np.median(counts))}, max {counts.max()}  (S = {a.shots} per point)')
    zero_count = int(np.sum(counts == 0))
    print(f'points with zero counts: {zero_count}  -> 95% upper limit 3/S = {3/a.shots:.1e} each')
    out = dict(L=a.L, M=a.M, phi=a.phi, shots=a.shots, floor=a.floor, signal=a.signal, halfwidth=h, aer_p2=max(a.aer_p2, a.dm_p2),
               exact_zero=[z0.real, z0.imag], all_exact_zeros=[[z.real, z.imag] for z in zeros],
               grid=pts, P_exact=Pex.tolist(), counts=counts.tolist(),
               fit=dict(a0=fit['z0'].real, theta0=fit['z0'].imag, G=fit['G'], B=fit['B']),
               error=err, ci_a=ci_a.tolist(), ci_theta=ci_t.tolist(), bootstrap_a=zb.real.tolist(), bootstrap_theta=zb.imag.tolist())
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    json.dump(out, open(a.out, 'w'), indent=1)
    print('->', a.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['demo'])
    ap.add_argument('--L', type=int, default=6); ap.add_argument('--M', type=int, default=3); ap.add_argument('--phi', type=float, default=0.6)
    ap.add_argument('--shots', type=int, default=200000); ap.add_argument('--floor', type=float, default=2e-4)
    ap.add_argument('--signal', type=float, default=0.5, help='multiplicative attenuation of the ideal probability by noise')
    ap.add_argument('--halfwidth', type=float, default=0.06); ap.add_argument('--nboot', type=int, default=200)
    ap.add_argument('--seed', type=int, default=1); ap.add_argument('--out', default='results/zero_localisation_demo.json')
    ap.add_argument('--aer_p2', type=float, default=0.0, help='if > 0: full Aer noisy LCU simulation with this CZ depolarising error')
    ap.add_argument('--aer_pro', type=float, default=0.01)
    ap.add_argument('--dm_p2', type=float, default=0.0, help='if > 0: exact density-matrix evaluation of the same circuit-level noise model (fast)')
    a = ap.parse_args()
    demo(a)


if __name__ == '__main__':
    main()
