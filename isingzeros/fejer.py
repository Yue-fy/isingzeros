"""
Off-axis zeros from data ON the unitary axis (no post-selection, no mid-circuit measurement).

Structure.  For a periodic chain of even length L (or W such legs), Sigma_k Z_k Z_{k+1} = L - 4q,
q = 0..L/2.  The strip amplitude with M layers V_y(K2) = exp(K2 Sigma ZZ) and theta-independent V_x is
therefore
        A(K2) = <+| V_y (V_x V_y)^{M-1} |+> = e^{E K2} F(u),   u = e^{-4 K2},   E = W L M,
        F(u)  = sum_{Q=0}^{n} c_Q u^Q,                         n = W L M / 2,
a polynomial of degree n in u.  The unitary family K2 = i theta is the unit circle |u| = 1, and a zero
K2 = a0 + i theta0 is a root u0 = e^{-4 a0} e^{-4 i theta0}:  |u0| < 1 for a0 > 0, |u0| > 1 for a0 < 0.

Fejer-Riesz.  Protocol A measures only P(theta) = |A(i theta)|^2 = |F(u)|^2 on |u| = 1, a real
trigonometric polynomial of degree n in u (2n+1 real parameters).  Its analytic extension
        G(u) = u^n sum_{k=-n}^{n} r_k u^k = F(u) F*(1/u) u^n
has as roots the n roots of F AND their reflections u -> 1/conj(u), i.e. (a0, theta0) -> (-a0, theta0).
Hence return probabilities alone determine every finite-size zero's |a0| and theta0; only the sign of a0
is lost (and a zero ON the axis appears as a double root).  The sign is fixed by a two-point post-selected
test (sign_test_*), not by a 2D grid.

Symmetries.  On a ring of even length, theta -> theta + pi/2 multiplies every V_y layer by the product of
its ZZ factors, which is the identity; the sublattice flip (X on odd sites) gives theta -> pi/2 - theta.
Hence P(theta) has period pi/2 and P(theta) = P(pi/2 - theta): the r_k are real, and only the n + 1 cosine
coefficients are free.  Both statements survive ANY Pauli noise channel (depolarising, twirled noise):
the extra Paulis commute through Pauli channels.  So under Pauli noise the measured P stays exactly in the
(n+1)-dimensional cosine space; sine terms and the theta-frequencies 4k+2 appear only through non-Pauli
errors (amplitude damping, coherent errors) or an angle-dependent compilation.  trig_fit(symmetric=True)
fits the cosine part and reports the sine part as a non-Pauli diagnostic.

Noise.  A constant floor B added to P changes only r_0; under a global-depolarising model
P_obs = alpha |A|^2 + B, alpha drops out of the roots and B must be subtracted (modes of `floor`).
Local gate noise and readout errors are not affine; their effect is quantified in verify.py (V3, V6).

    python -m isingzeros.fejer demo --L 6 --M 3 --phi 0.6
"""
from __future__ import annotations
import argparse, json
import numpy as np
from .classical import RowTransfer, primal_coupling


# ----------------------------------------------------------------------------- structure
def degree(L, M, W=1):
    if L % 2:
        raise ValueError('Fejer-Riesz structure needs even L (periodic legs: Sigma ZZ = L - 4q)')
    return W * L * M // 2


def theta_grid(n, K=None, offset=None, period=np.pi / 2):
    """K uniform points on [0, period), by default shifted by half a step.  period = pi/2 is one period of
    |A|^2 (u = e^{-4 i theta} goes once round the circle).  The half-step shift keeps the grid off theta = 0 and
    pi/4, where RZZ(-2 theta) is the identity or a Clifford gate and a compiler may simplify the circuit; the
    shifted grid is still symmetric under theta -> pi/2 - theta."""
    K = int(K or (4 * n + 4))
    offset = 0.5 * period / K if offset is None else offset
    return offset + period * np.arange(K) / K


def z_from_u(u):
    """u = exp(-4 K2)  ->  K2 = a + i theta with theta in [0, pi/2)."""
    K2 = -np.log(np.asarray(u, complex)) / 4
    return K2.real + 1j * np.mod(K2.imag, np.pi / 2)


def u_from_z(z):
    return np.exp(-4 * np.asarray(z, complex))


def amplitude(L, M, phi, K2, W=1, theta3=0.0):
    return RowTransfer(L, primal_coupling(1j * phi), K2, W=W, K3=1j * theta3).strip_amplitude(M)


def return_probabilities_exact(L, M, phi, thetas, W=1, theta3=0.0):
    return np.array([abs(amplitude(L, M, phi, 1j * t, W, theta3)) ** 2 for t in thetas])


# ----------------------------------------------------------------------------- exact (complex) route
def exact_poly_coefficients(L, M, phi, W=1, theta3=0.0):
    """c_Q from complex amplitudes on the unit circle (classical reference; on hardware this would need a
    Hadamard test)."""
    n = degree(L, M, W); E = W * L * M
    th = theta_grid(n, 2 * n + 2)
    A = np.array([amplitude(L, M, phi, 1j * t, W, theta3) for t in th])
    y = A * np.exp(-1j * E * th)
    V = u_from_z(1j * th)[:, None] ** np.arange(n + 1)[None, :]
    c, *_ = np.linalg.lstsq(V, y, rcond=None)
    return c


A_CAP = 4.0      # |Re K2| beyond which a root is treated as a zero 'at infinity' (degree deficiency of F)


def exact_zeros(L, M, phi, W=1, theta3=0.0):
    """All finite-size zeros K2 = a0 + i theta0 (theta0 in [0, pi/2)) from the exact polynomial F.  At special
    points (e.g. phi = pi/4) the leading or trailing coefficients of F vanish; the corresponding roots at
    u = 0 or u = infinity (Re K2 = +-infinity) are dropped."""
    c = exact_poly_coefficients(L, M, phi, W, theta3)
    big = np.abs(c) > 1e-11 * np.abs(c).max()
    c = c[np.min(np.nonzero(big)): np.max(np.nonzero(big)) + 1]
    z = z_from_u(np.roots(c[::-1])) if len(c) > 1 else np.array([], complex)
    return np.sort_complex(z[np.abs(z.real) < A_CAP])


# ----------------------------------------------------------------------------- Fejer-Riesz route
def trig_fit(thetas, P, n, sigma=None, nuisance=False, return_nuisance=False, symmetric=False):
    """Weighted least squares  P(theta) = sum_{k=-n}^{n} r_k u^k  (r_{-k} = conj r_k).  Returns r[k+n].
    symmetric=True keeps only the cosine part (real r_k), which is all that ideal or Pauli-noisy data contain;
    the fitted sine part is discarded and its norm returned as a non-Pauli diagnostic (return_nuisance=True).
    nuisance=True also fits the theta-frequencies 2, 6, ..., 4n-2 and discards them (needs thetas covering
    [0, pi)); they too vanish under Pauli noise."""
    th = np.asarray(thetas); P = np.asarray(P, float)
    cols = [np.ones_like(th)]
    for k in range(1, n + 1):
        cols.append(2 * np.cos(4 * k * th)); cols.append(2 * np.sin(4 * k * th))
    nall = len(cols)
    if nuisance:
        for k in range(n):
            cols.append(2 * np.cos((4 * k + 2) * th)); cols.append(2 * np.sin((4 * k + 2) * th))
    X = np.stack(cols, 1)
    w = np.ones_like(th) if sigma is None else 1.0 / np.asarray(sigma)
    beta, *_ = np.linalg.lstsq(X * w[:, None], P * w, rcond=None)
    nuis2 = float(np.sum(beta[nall:] ** 2) * 2) if nuisance else 0.0
    beta = beta[:nall].copy()
    if symmetric:
        nuis2 += float(np.sum(beta[2::2] ** 2) * 2)
        beta[2::2] = 0.0
    nuis = float(np.sqrt(nuis2))
    r = np.zeros(2 * n + 1, complex)
    r[n] = beta[0]
    for k in range(1, n + 1):
        r[n + k] = beta[2 * k - 1] + 1j * beta[2 * k]
        r[n - k] = np.conj(r[n + k])
    return (r, nuis) if return_nuisance else r


def trig_eval(r, thetas):
    n = (len(r) - 1) // 2
    u = u_from_z(1j * np.asarray(thetas))
    return np.real(sum(r[n + k] * u ** k for k in range(-n, n + 1)))


def pair_roots(roots, tol=1e-9):
    """Group the 2n roots of G into n mirror pairs {u, 1/conj(u)} (greedy matching).  Each pair ->
    dict(theta, a_abs, u_in (|u|<=1, a>=0 branch), u_out)."""
    roots = list(roots)
    used = [False] * len(roots)
    pairs = []
    mir = [1 / np.conj(u) for u in roots]
    D = np.array([[abs(roots[j] - mir[i]) if i != j else np.inf for j in range(len(roots))] for i in range(len(roots))])
    order = np.dstack(np.unravel_index(np.argsort(D, axis=None), D.shape))[0]
    for i, j in order:
        if used[i] or used[j]:
            continue
        used[i] = used[j] = True
        zi, zj = z_from_u(roots[i]), z_from_u(roots[j])
        # circular mean of theta with period pi/2
        ang = np.angle(np.exp(4j * zi.imag) + np.exp(4j * zj.imag)) / 4
        th = np.mod(ang, np.pi / 2)
        a_abs = abs(zi.real - zj.real) / 2
        if abs(abs(roots[i]) - 1) < 1e-9 and abs(abs(roots[j]) - 1) < 1e-9:
            a_abs = 0.0             # tangential split of a (near-)double root on the circle: compatible with a = 0
        ui, uo = (roots[i], roots[j]) if abs(roots[i]) <= abs(roots[j]) else (roots[j], roots[i])
        pairs.append(dict(theta=float(th), a_abs=float(a_abs), u_in=complex(ui), u_out=complex(uo),
                          tangential=float(abs(np.angle(np.exp(4j * (zi.imag - zj.imag)))) / 4)))
        if all(used):
            break
    return sorted([p for p in pairs if p['a_abs'] < A_CAP], key=lambda p: p['theta'])


def floor_value(r, mode, theta_star=None, fine=4000):
    if mode in (None, 'none'):
        return 0.0
    if isinstance(mode, (int, float)):
        return float(mode)
    if mode == 'onaxis':          # value of the fitted polynomial at a known exact on-axis zero
        return float(trig_eval(r, [theta_star])[0])
    if mode == 'min':             # largest floor compatible with P >= 0 (forces a double root)
        return float(trig_eval(r, np.linspace(0, np.pi / 2, fine, endpoint=False)).min())
    raise ValueError(mode)


def reconstruct(thetas, P, n, sigma=None, floor='none', theta_star=None, nuisance=False, symmetric=False):
    """Return probabilities on the unit circle -> pairs (|a0|, theta0).  floor: 'none' | 'onaxis' | 'min' | float."""
    r = trig_fit(thetas, P, n, sigma, nuisance, symmetric=symmetric)
    B = floor_value(r, floor, theta_star)
    r = r.copy(); r[n] -= B
    roots = np.roots(r[::-1])
    roots = roots[(np.abs(roots) > 0) & np.isfinite(roots)]
    return pair_roots(roots), r, B


def match(pairs, zeros):
    """For each exact zero (a0 + i theta0) the reconstructed pair closest in (|a|, theta)."""
    out = []
    for z0 in zeros:
        d = [np.hypot(p['a_abs'] - abs(z0.real), abs(np.angle(np.exp(4j * (p['theta'] - z0.imag)))) / 4) for p in pairs]
        j = int(np.argmin(d))
        out.append(dict(a0=float(z0.real), theta0=float(z0.imag), a_abs_hat=pairs[j]['a_abs'], theta_hat=pairs[j]['theta'],
                        err_a=pairs[j]['a_abs'] - abs(z0.real),
                        err_theta=float(np.angle(np.exp(4j * (pairs[j]['theta'] - z0.imag))) / 4), dist=float(d[j]), index=j))
    return out


def sample_and_reconstruct(thetas, P_true, shots, n, rng, floor='none', theta_star=None, nuisance=False, symmetric=False):
    k = rng.binomial(shots, np.clip(P_true, 0, 1))
    Ph = k / shots
    sigma = np.sqrt(np.maximum(Ph * (1 - Ph), 1.0 / shots) / shots)
    pairs, r, B = reconstruct(thetas, Ph, n, sigma, floor, theta_star, nuisance, symmetric)
    return pairs, r, B, k


def bootstrap(thetas, counts, shots, n, nboot=200, rng=None, floor='none', theta_star=None, nuisance=False, symmetric=False):
    """Parametric bootstrap of the reconstructed pairs: resample counts from the fitted (floored back) model."""
    rng = np.random.default_rng(rng)
    Ph = np.asarray(counts) / shots
    sigma = np.sqrt(np.maximum(Ph * (1 - Ph), 1.0 / shots) / shots)
    pairs0, r0, B0 = reconstruct(thetas, Ph, n, sigma, floor, theta_star, nuisance, symmetric)
    r_model = r0.copy(); r_model[n] += B0
    Pm = np.clip(trig_eval(r_model, thetas), 0, 1) if not nuisance else np.clip(Ph, 0, 1)
    ref = [p['a_abs'] + 1j * p['theta'] for p in pairs0]
    samples = []
    for _ in range(nboot):
        pb, *_ = sample_and_reconstruct(thetas, Pm, shots, n, rng, floor, theta_star, nuisance, symmetric)
        m = match(pb, [complex(z.real, z.imag) for z in ref])
        samples.append([(x['a_abs_hat'], x['theta_hat']) for x in m])
    return pairs0, np.array(samples)          # samples[b, pair, (|a|, theta)]


# ----------------------------------------------------------------------------- sign of a0
def sign_test_points(pair):
    """Mirror points t+ = (+|a|, theta), t- = (-|a|, theta).  Under H+ (zero at +|a|) P_joint(t+) = 0 < P_joint(t-);
    under H- the opposite.  Decision: sign of p_hat(t-) - p_hat(t+); an affine noise model P -> alpha P + B
    (alpha > 0) does not change the sign."""
    return (pair['a_abs'], pair['theta']), (-pair['a_abs'], pair['theta'])


def sign_test_shots(p_plus, p_minus, eps=1e-3):
    """Shots per point for a one-sided error probability eps (normal approximation of the difference)."""
    from scipy.stats import norm
    z = norm.ppf(1 - eps)
    d = abs(p_plus - p_minus)
    return np.inf if d == 0 else float(np.ceil(z ** 2 * (p_plus + p_minus) / d ** 2))


def demo(a):
    from .analysis import joint_probability
    n = degree(a.L, a.M)
    zs = exact_zeros(a.L, a.M, a.phi)
    th = theta_grid(n, a.K)
    P = return_probabilities_exact(a.L, a.M, a.phi, th)
    pairs, r, B = reconstruct(th, P, n)
    print(f'L={a.L} M={a.M} phi={a.phi}: degree n={n}, {len(th)} theta points')
    for m in match(pairs, zs):
        print(f"  exact zero a0={m['a0']:+.6f} theta0={m['theta0']:.6f}   from |A|^2 only: |a|={m['a_abs_hat']:.6f} theta={m['theta_hat']:.6f}")
    for p in pairs:
        if p['a_abs'] > 1e-6 and abs(p['a_abs']) < 0.3:
            tp, tm = sign_test_points(p)
            pp, pm = joint_probability(a.L, a.M, a.phi, *tp), joint_probability(a.L, a.M, a.phi, *tm)
            print(f"  sign test at theta={p['theta']:.4f}: P_joint(+|a|)={pp:.2e}, P_joint(-|a|)={pm:.2e}, "
                  f"shots/point for 1e-3 error: {sign_test_shots(pp, pm):.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['demo'])
    ap.add_argument('--L', type=int, default=6); ap.add_argument('--M', type=int, default=3)
    ap.add_argument('--phi', type=float, default=0.6); ap.add_argument('--K', type=int, default=None)
    a = ap.parse_args()
    demo(a)


if __name__ == '__main__':
    main()
