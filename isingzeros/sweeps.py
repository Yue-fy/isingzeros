"""
Parameter sweeps (the HPC workload).  Every sweep writes one JSON file; plots.py reads them.

    python -m isingzeros.sweeps exact   --L 8 --M 4 --W 1 --ntheta 121 --nphi 121 --out results/exact_L8M4.json
    python -m isingzeros.sweeps zeros   --L 8 --M 4 --phi 0.6 --out results/zeros_L8M4_phi0.6.json
    python -m isingzeros.sweeps locus   --out results/locus.json
    python -m isingzeros.sweeps noisy   --L 8 --M 4 --phi 0.6 --ntheta 41 --shots 20000 --p2 0.005 --out results/noisy_L8M4.json
    python -m isingzeros.sweeps lcu     --L 6 --M 3 --phi 0.6 --theta 0.7 --a_zz 0.0:0.3:7 --shots 40000 --out results/lcu_L6M3.json
    python -m isingzeros.sweeps budget  --L 8 --M 4 --out results/budget.json

"exact": return probability |<+|U|+>|^2 on a (theta, phi) grid from the transfer matrix (no Qiskit).
"zeros": zeros of the strip partition function in the complex theta plane at fixed phi (theta = Im K2
         on the real axis of this plane, Re theta... see classical.find_zeros_on_grid), plus the
         predicted thermodynamic locus (Onsager-Kaufman factor).
"noisy": Aer simulation with a generic noise model along a theta line, compared with exact values.
"lcu":   complex couplings off the imaginary axis via post-selected LCU (Aer), compared with exact.
"budget": two-qubit gate counts, depths and shot budgets for the sizes planned on hardware.
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
from multiprocessing import Pool

from .classical import RowTransfer, primal_coupling, circuit_couplings, onsager_kaufman_min, find_zeros_on_grid


def _parse_range(s: str):
    a, b, n = s.split(':'); return np.linspace(float(a), float(b), int(n))


def _amp(args):
    L, M, W, theta, phi, theta3 = args
    K1 = primal_coupling(1j * phi); K2 = 1j * theta; K3 = 1j * theta3
    return RowTransfer(L, K1, K2, W=W, K3=K3).strip_amplitude(M)


def sweep_exact(a):
    thetas = np.linspace(0.0, np.pi / 2, a.ntheta); phis = np.linspace(0.02, np.pi / 2 - 0.02, a.nphi)
    jobs = [(a.L, a.M, a.W, th, ph, a.theta3) for th in thetas for ph in phis]
    t0 = time.time()
    with Pool(a.nproc) as pool:
        amps = pool.map(_amp, jobs, chunksize=8)
    amps = np.array(amps).reshape(a.ntheta, a.nphi)
    out = {'L': a.L, 'M': a.M, 'W': a.W, 'theta3': a.theta3, 'theta': thetas.tolist(), 'phi': phis.tolist(),
           'return_prob': (np.abs(amps) ** 2).tolist(), 'amp_re': amps.real.tolist(), 'amp_im': amps.imag.tolist(),
           'seconds': time.time() - t0}
    json.dump(out, open(a.out, 'w'))
    print(f'exact sweep done in {out["seconds"]:.1f}s -> {a.out}')


def sweep_zeros(a):
    """Zeros of the strip partition function in the complex theta plane (theta -> theta_r + i theta_i,
    i.e. K2 = -theta_i + i theta_r: the imaginary part of theta is a REAL coupling).  phi fixed."""
    K1 = primal_coupling(1j * a.phi)
    pref = lambda M, n: 1.0   # zeros do not depend on the prefactor (2 sinh 2K1)^... which is theta-independent

    def Zs(th):
        T = RowTransfer(a.L, K1, 1j * th, W=a.W, K3=1j * a.theta3)
        return T.strip_amplitude(a.M)
    zeros = find_zeros_on_grid(Zs, (0.0, np.pi / 2), (-a.im_max, a.im_max), nre=a.nre, nim=a.nim)
    # predicted locus: points (theta_r, theta_i) where the Onsager-Kaufman factor can vanish for real momenta
    thr = np.linspace(0.0, np.pi / 2, 91); thi = np.linspace(-a.im_max, a.im_max, 61)
    okm = np.array([[onsager_kaufman_min(K1, 1j * (x + 1j * y)) for x in thr] for y in thi])
    out = {'L': a.L, 'M': a.M, 'W': a.W, 'phi': a.phi, 'K1': [K1.real, K1.imag],
           'zeros': [[z.real, z.imag] for z in zeros],
           'locus_theta_r': thr.tolist(), 'locus_theta_i': thi.tolist(), 'ok_min': okm.tolist()}
    json.dump(out, open(a.out, 'w'))
    print(f'{len(zeros)} zeros found -> {a.out}')


def sweep_locus(a):
    """The predicted zero locus on the unitary family (theta, phi) (imaginary couplings only)."""
    thetas = np.linspace(0.0, np.pi / 2, a.ntheta); phis = np.linspace(0.02, np.pi / 2 - 0.02, a.nphi)
    okm = np.array([[onsager_kaufman_min(*circuit_couplings(th, ph)) for ph in phis] for th in thetas])
    json.dump({'theta': thetas.tolist(), 'phi': phis.tolist(), 'ok_min': okm.tolist()}, open(a.out, 'w'))
    print(f'locus -> {a.out}')


def sweep_noisy(a):
    from .simulate import return_probability, generic_noise, exact_amplitude
    thetas = np.linspace(0.0, np.pi / 2, a.ntheta)
    noise = generic_noise(p2=a.p2, p1=a.p1, p_ro=a.p_ro)
    rows = []
    for th in thetas:
        ex = abs(exact_amplitude(a.L, a.M, th, a.phi, a.W, a.theta3)) ** 2
        p_ideal, t = return_probability(a.L, a.M, th, a.phi, a.W, a.theta3, shots=a.shots, noise=None, seed=a.seed)
        p_noisy, _ = return_probability(a.L, a.M, th, a.phi, a.W, a.theta3, shots=a.shots, noise=noise, seed=a.seed)
        rows.append({'theta': th, 'exact': ex, 'ideal_shots': p_ideal, 'noisy': p_noisy})
        print(f'theta={th:.3f} exact={ex:.4f} ideal={p_ideal:.4f} noisy={p_noisy:.4f}')
    ops = dict(t.count_ops())
    json.dump({'L': a.L, 'M': a.M, 'W': a.W, 'phi': a.phi, 'shots': a.shots, 'p2': a.p2, 'p1': a.p1, 'p_ro': a.p_ro,
               'two_qubit_gates': ops.get('cz', 0), 'depth': t.depth(), 'rows': rows}, open(a.out, 'w'))
    print(f'noisy sweep -> {a.out}')


def sweep_lcu(a):
    from .simulate import lcu_return_probability, generic_noise, exact_amplitude
    noise = generic_noise(p2=a.p2, p1=a.p1, p_ro=a.p_ro) if a.p2 > 0 else None
    rows = []
    for az in _parse_range(a.a_zz):
        ex = abs(exact_amplitude(a.L, a.M, a.theta, a.phi, a.W, a.theta3, a_zz=az, b_x=a.b_x)) ** 2
        est, psucc, t = lcu_return_probability(a.L, a.M, a.theta, a.phi, az, a.b_x, a.W, a.theta3, shots=a.shots, noise=noise, seed=a.seed)
        rows.append({'a_zz': float(az), 'exact': ex, 'estimate': est, 'p_success': psucc})
        print(f'a_zz={az:.3f} exact={ex:.4f} est={est:.4f} p_succ={psucc:.3f}')
    ops = dict(t.count_ops())
    json.dump({'L': a.L, 'M': a.M, 'W': a.W, 'theta': a.theta, 'phi': a.phi, 'b_x': a.b_x, 'shots': a.shots,
               'two_qubit_gates': ops.get('cz', 0) + ops.get('cx', 0), 'depth': t.depth(), 'rows': rows}, open(a.out, 'w'))
    print(f'lcu sweep -> {a.out}')


def sweep_budget(a):
    """Gate counts and shot budgets for the hardware plan."""
    from .circuits import kicked_ising_unitary, kicked_ising_hadamard_test, kicked_ising_lcu, gate_counts
    plan = []
    for (L, M, W) in [(8, 3, 1), (8, 4, 1), (12, 4, 1), (16, 4, 1), (16, 6, 1), (24, 4, 1), (8, 3, 2), (12, 4, 2), (16, 3, 2), (8, 3, 3)]:
        g = gate_counts(kicked_ising_unitary(L, M, 0.4, 0.4, W=W, theta3=0.3 if W > 1 else 0.0))
        # typical return probability scale: exp(-2 n lambda), lambda ~ 0.1-0.3 per site at a zero crossing region
        n = L * W
        p_typ = np.exp(-2 * n * 0.1)
        shots_10pct = int(100 / p_typ)          # relative error 10% at p_typ needs ~100/p shots
        plan.append({'L': L, 'M': M, 'W': W, 'qubits': n, 'two_qubit': g['two_qubit'], 'depth': g['depth'],
                     'p_typical': p_typ, 'shots_for_10pct': shots_10pct})
    small = []
    for (L, M) in [(4, 3), (6, 3), (8, 3)]:
        g = gate_counts(kicked_ising_hadamard_test(L, M, 0.4, 0.4))
        gl = gate_counts(kicked_ising_lcu(L, M, 0.4, 0.4, 0.15, 0.0))
        # exact LCU success probability at theta=phi=0.4, a_zz=0.15: N^2 ||U|+>||^2
        from .circuits import lcu_normalisation
        T = RowTransfer(L, primal_coupling(1j * 0.4), 0.15 + 0.4j)
        psucc = lcu_normalisation(L, M, 0.15, 0.0) ** 2 * float(np.vdot(T.strip_vector(M), T.strip_vector(M)).real)
        small.append({'L': L, 'M': M, 'hadamard_two_qubit': g['two_qubit'], 'hadamard_depth': g['depth'],
                      'lcu_two_qubit': gl['two_qubit'], 'lcu_depth': gl['depth'],
                      'lcu_success_prob_a0.15': psucc, 'lcu_lower_bound': float(np.exp(-2 * 0.15 * M * L))})
    json.dump({'unitary_plan': plan, 'small_protocols': small}, open(a.out, 'w'), indent=1)
    for p in plan:
        print(p)
    for s in small:
        print(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['exact', 'zeros', 'locus', 'noisy', 'lcu', 'budget'])
    ap.add_argument('--L', type=int, default=8); ap.add_argument('--M', type=int, default=4); ap.add_argument('--W', type=int, default=1)
    ap.add_argument('--theta3', type=float, default=0.0)
    ap.add_argument('--ntheta', type=int, default=61); ap.add_argument('--nphi', type=int, default=61)
    ap.add_argument('--phi', type=float, default=0.6); ap.add_argument('--theta', type=float, default=0.7)
    ap.add_argument('--im_max', type=float, default=0.6); ap.add_argument('--nre', type=int, default=121); ap.add_argument('--nim', type=int, default=81)
    ap.add_argument('--shots', type=int, default=20000); ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--p2', type=float, default=5e-3); ap.add_argument('--p1', type=float, default=5e-4); ap.add_argument('--p_ro', type=float, default=1e-2)
    ap.add_argument('--a_zz', type=str, default='0.0:0.3:7'); ap.add_argument('--b_x', type=float, default=0.0)
    ap.add_argument('--nproc', type=int, default=max(1, os.cpu_count() or 1))
    ap.add_argument('--out', type=str, required=True)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    {'exact': sweep_exact, 'zeros': sweep_zeros, 'locus': sweep_locus, 'noisy': sweep_noisy, 'lcu': sweep_lcu, 'budget': sweep_budget}[a.mode](a)


if __name__ == '__main__':
    main()
