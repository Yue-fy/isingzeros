"""pytest -q  (takes ~1 min).  Cross-validation: brute force = transfer matrix = statevector circuit = shots."""
import numpy as np, itertools, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from isingzeros.classical import RowTransfer, Z_bruteforce, primal_coupling, dual_coupling, onsager_kaufman_min
from isingzeros.simulate import exact_amplitude, statevector_amplitude, return_probability, hadamard_amplitude, lcu_return_probability


def test_duality():
    for K in [0.3, 0.7 + 0.2j, -0.1 + 1.1j]:
        assert abs(primal_coupling(dual_coupling(K)) - K) < 1e-12


def test_transfer_matrix_vs_bruteforce():
    K1, K2 = 0.4 + 0.25j, 0.3 - 0.2j
    for (L, M) in [(3, 3), (4, 3), (3, 4)]:
        T = RowTransfer(L, K1, K2)
        for torus in (True, False):
            zb = Z_bruteforce(L, M, K1, K2, torus=torus)
            zt = T.Z_torus(M) if torus else T.Z_strip(M)
            assert abs(zt - zb) / abs(zb) < 1e-12


def test_ladder_vs_bruteforce():
    L, M, W, K1, K2, K3 = 2, 3, 2, 0.4 + 0.25j, 0.3 - 0.2j, 0.3 + 0.1j
    T = RowTransfer(L, K1, K2, W=W, K3=K3)
    Z = 0
    for bits in itertools.product((1, -1), repeat=W * L * M):
        s = np.array(bits).reshape(M, W, L)
        e = K2 * np.sum(s * np.roll(s, -1, axis=2)) + K3 * np.sum(s[:, :-1, :] * s[:, 1:, :]) + K1 * np.sum(s[:-1] * s[1:])
        Z += np.exp(e)
    assert abs(T.Z_strip(M) - Z) / abs(Z) < 1e-12


def test_circuit_vs_transfer_matrix():
    for (L, M, th, ph, W, th3) in [(4, 2, 0.3, 0.5, 1, 0), (5, 3, 0.7, 0.4, 1, 0), (3, 3, 0.3, 0.6, 2, 0.25)]:
        assert abs(exact_amplitude(L, M, th, ph, W, th3) - statevector_amplitude(L, M, th, ph, W, th3)) < 1e-12


def test_shots():
    p, _ = return_probability(4, 3, 0.3, 0.5, shots=20000)
    assert abs(p - abs(exact_amplitude(4, 3, 0.3, 0.5)) ** 2) < 0.02
    h, _ = hadamard_amplitude(4, 2, 0.3, 0.5, shots=20000)
    assert abs(h - exact_amplitude(4, 2, 0.3, 0.5)) < 0.03


def test_lcu():
    r, ps, _ = lcu_return_probability(3, 2, 0.3, 0.5, 0.15, 0.1, shots=40000)
    ex = abs(exact_amplitude(3, 2, 0.3, 0.5, a_zz=0.15, b_x=0.1)) ** 2
    assert abs(r - ex) / ex < 0.1


def test_isotropic_circle():
    Kc = 0.5 * np.log(1 + np.sqrt(2))      # sinh 2Kc = 1
    assert onsager_kaufman_min(Kc, Kc) < 1e-8
    assert onsager_kaufman_min(0.3, 0.3) > 0.1


def test_lcu_negative_and_deferred():
    from isingzeros.circuits import kicked_ising_lcu, lcu_postselect, lcu_normalisation
    from isingzeros.simulate import run_counts
    for a in (0.15, -0.15):
        ex = abs(exact_amplitude(3, 2, 0.3, 0.5, a_zz=a, b_x=-0.1)) ** 2
        r, ps, _ = lcu_return_probability(3, 2, 0.3, 0.5, a, -0.1, shots=60000)
        assert abs(r - ex) / ex < 0.1
        qc = kicked_ising_lcu(3, 2, 0.3, 0.5, a, -0.1, deferred=True)
        counts, _ = run_counts(qc, 60000, None, 3)
        p_succ, p_ret = lcu_postselect(counts, 3)
        assert abs(p_ret * p_succ / lcu_normalisation(3, 2, a, -0.1) ** 2 - ex) / ex < 0.1


def test_noisy_joint_probability_dm():
    """Density-matrix evaluation of the circuit-level noise model: noiseless limit = exact joint probability;
    noisy value agrees with an Aer trajectory simulation within statistics."""
    from isingzeros.analysis import noisy_joint_probability_dm, joint_probability
    from isingzeros.circuits import kicked_ising_lcu, lcu_postselect
    from isingzeros.simulate import run_counts, generic_noise
    qc = kicked_ising_lcu(4, 2, 0.8, 0.6, -0.1, 0.0)
    ex = joint_probability(4, 2, 0.6, -0.1, 0.8)
    assert abs(noisy_joint_probability_dm(qc, 0, 0, 0) - ex) < 1e-10
    p = noisy_joint_probability_dm(qc, 5e-3, 5e-4, 1e-2)
    c, _ = run_counts(qc, 100000, generic_noise(5e-3, 5e-4, 1e-2), 3)
    ps, pr = lcu_postselect(c, 4)
    assert abs(ps * pr - p) < 4 * np.sqrt(p / 100000)


def test_fejer_riesz_noiseless():
    """Zeros from return probabilities only (|A|^2 on the unitary axis) = exact zeros up to the sign of Re K2;
    the exact polynomial zeros agree with the transfer-matrix grid search."""
    from isingzeros import fejer as fr
    from isingzeros.analysis import exact_zeros_K2plane
    for (L, M, phi) in [(4, 2, 0.6), (6, 3, 0.6), (4, 2, 0.4)]:
        n = fr.degree(L, M)
        zs = fr.exact_zeros(L, M, phi)
        for full in (False, True):
            th = fr.theta_grid(n, 80 if full else None, period=np.pi if full else np.pi / 2)
            pairs, _, _ = fr.reconstruct(th, fr.return_probabilities_exact(L, M, phi, th), n, nuisance=full)
            for m in fr.match(pairs, zs):
                assert abs(m['err_a']) < 1e-6 and abs(m['err_theta']) < 1e-6
        for g in exact_zeros_K2plane(L, M, phi, a_max=0.2):
            assert min(abs(g - z) for z in zs) < 1e-6


def test_compile_fixed_angle_independent():
    """The noisy simulations compile every angle as a parameter: same gate sequence at theta = 0, pi/4, generic,
    and the same LCU structure at a = 0 and a != 0 (no special-casing of a = 0)."""
    from isingzeros.circuits import kicked_ising_unitary, kicked_ising_lcu, compile_fixed
    counts = {dict(compile_fixed(kicked_ising_unitary(4, 2, t, 0.6)).count_ops()).get('cz', 0) for t in (0.0, np.pi / 4, np.pi / 2, 0.37)}
    assert counts == {16}
    lcu = {tuple(sorted(dict(compile_fixed(kicked_ising_lcu(4, 2, 0.7, 0.6, a, 0.0, reset=False)).count_ops()).items())) for a in (0.0, 0.05)}
    assert len(lcu) == 1


def test_pauli_noise_keeps_symmetries():
    """Under depolarising (Pauli) noise P(theta) keeps period pi/2 and P(theta) = P(pi/2 - theta) exactly."""
    from isingzeros.analysis import noisy_joint_probability_dm
    from isingzeros.circuits import kicked_ising_unitary
    P = lambda t: noisy_joint_probability_dm(kicked_ising_unitary(4, 2, t, 0.6), 0.01, 0.001, 0.02)
    for t in (0.13, 0.52):
        assert abs(P(t) - P(t + np.pi / 2)) < 1e-12
        assert abs(P(t) - P(np.pi / 2 - t)) < 1e-12
