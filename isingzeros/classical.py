"""
Classical side: partition functions of the anisotropic 2D Ising model at COMPLEX couplings,
by brute force (tiny lattices, for testing) and by the row transfer matrix (dense vectors, L <= ~24).

Conventions:
    - lattice sites (k, m): k = 0..L-1 along a periodic row (horizontal), m = row index (vertical);
    - horizontal bonds carry K2, vertical bonds carry K1 (couplings in units of k_B T, complex allowed);
    - Boltzmann weight exp( K2 * sum_{m,k} s_{m,k} s_{m,k+1} + K1 * sum_{m,k} s_{m,k} s_{m+1,k} ).

Transfer matrix (row-to-row, acting on the 2^L-dimensional row space):
    V_y = exp( K2 * sum_k Z_k Z_{k+1} )           (diagonal)
    V_x = exp( K1* * sum_k X_k ),  tanh K1* = exp(-2 K1)   (dual coupling)
    exp(K1 s s') = (2 sinh 2K1)^{1/2} <s| exp(K1* X) |s'>      (the standard identity)

    torus  (L x M, periodic in both directions):
        Z_torus = (2 sinh 2K1)^{L M / 2} * Tr[ (V_x V_y)^M ]
    strip  (L x M, periodic in the row direction, FREE at the first and last row):
        Z_strip = (2 sinh 2K1)^{L (M-1) / 2} * 2^L * <+^L| V_y (V_x V_y)^{M-1} |+^L>
    where |+^L> = 2^{-L/2} sum_s |s> is the x-polarised product state.

The strip formula is the one realised on a quantum computer: for purely imaginary K2 = i*theta and
K1* = i*phi the operator V_y (V_x V_y)^{M-1} is a kicked-Ising Floquet unitary, and
|<+|U|+>|^2 is a return probability (Loschmidt echo).  See circuits.py.
"""
from __future__ import annotations
import itertools
import numpy as np

# ----------------------------------------------------------------------------- couplings
def dual_coupling(K1: complex) -> complex:
    """K1* with tanh K1* = exp(-2 K1)  (principal branch of artanh)."""
    return np.arctanh(np.exp(-2.0 * K1))


def primal_coupling(K1s: complex) -> complex:
    """Inverse of dual_coupling: K1 = -1/2 log(tanh K1*)."""
    return -0.5 * np.log(np.tanh(K1s))


def sinh2(K: complex) -> complex:
    return np.sinh(2.0 * K)


# ----------------------------------------------------------------------------- brute force (tests)
def Z_bruteforce(L: int, M: int, K1: complex, K2: complex, torus: bool = True) -> complex:
    """Direct sum over all 2^(L M) configurations (use only for L*M <= 16)."""
    n = L * M
    if n > 20:
        raise ValueError("brute force limited to L*M <= 20")
    Z = 0.0 + 0.0j
    for bits in itertools.product((1, -1), repeat=n):
        s = np.array(bits).reshape(M, L)
        e = K2 * np.sum(s * np.roll(s, -1, axis=1))           # horizontal, periodic row
        if torus:
            e += K1 * np.sum(s * np.roll(s, -1, axis=0))      # vertical, periodic
        else:
            e += K1 * np.sum(s[:-1] * s[1:])                  # vertical, free ends
        Z += np.exp(e)
    return Z


# ----------------------------------------------------------------------------- transfer matrix on vectors
class RowTransfer:
    """Row transfer matrix of the L-site periodic row, applied to dense vectors of length 2^L.

    Basis: computational basis |s_0 ... s_{L-1}>, bit b_k = 0 <-> s_k = +1, b_k = 1 <-> s_k = -1.
    Also supports a ladder of W legs (W*L sites per row, rung coupling K3), i.e. a layered 3D model:
    the "row" is then a W x L sheet, and the transfer direction is the third dimension.
    """

    def __init__(self, L: int, K1: complex, K2: complex, W: int = 1, K3: complex = 0.0,
                 periodic_legs: bool = False):
        self.L, self.W = L, W
        self.n = L * W
        self.K1, self.K2, self.K3 = complex(K1), complex(K2), complex(K3)
        self.K1s = dual_coupling(self.K1)
        self.periodic_legs = periodic_legs
        N = 2 ** self.n
        idx = np.arange(N, dtype=np.int64)

        def spin(q):                                   # +1 / -1 for qubit q = w*L + k, int8 (N bytes)
            return (1 - 2 * ((idx >> q) & 1)).astype(np.int8)

        e = np.zeros(N, dtype=complex)
        for w in range(W):                             # leg bonds, periodic in L
            for k in range(L):
                e += self.K2 * (spin(w * L + k) * spin(w * L + (k + 1) % L))
        if W > 1:                                      # rungs
            wr = range(W) if periodic_legs else range(W - 1)
            for w in wr:
                for k in range(L):
                    e += self.K3 * (spin(w * L + k) * spin(((w + 1) % W) * L + k))
        self.diag_y = np.exp(e)                                      # V_y diagonal
        # V_x = prod_k exp(K1* X_k): 2x2 matrix per site
        c, sh = np.cosh(self.K1s), np.sinh(self.K1s)
        self.x2 = np.array([[c, sh], [sh, c]], dtype=complex)

    def apply_Vy(self, v: np.ndarray) -> np.ndarray:
        return self.diag_y * v

    def apply_Vx(self, v: np.ndarray) -> np.ndarray:
        """prod_k exp(K1* X_k) v, site by site, in place on a copy (bit k <-> axis of size 2)."""
        c, sh = self.x2[0, 0], self.x2[0, 1]
        w = v.astype(complex, copy=True)
        for k in range(self.n):
            t = w.reshape(2 ** (self.n - k - 1), 2, 2 ** k)      # index = high * 2^(k+1) + bit_k * 2^k + low
            t0 = t[:, 0, :].copy()
            t[:, 0, :] = c * t0 + sh * t[:, 1, :]
            t[:, 1, :] = sh * t0 + c * t[:, 1, :]
        return w

    def apply_T(self, v: np.ndarray) -> np.ndarray:
        """V_x V_y applied to v."""
        return self.apply_Vx(self.apply_Vy(v))

    # ------------------------------------------------------------------ partition functions
    def strip_vector(self, M: int) -> np.ndarray:
        """V_y (V_x V_y)^{M-1} |+^n>."""
        N = 2 ** self.n
        v = np.full(N, 1.0 / np.sqrt(N), dtype=complex)
        w = self.apply_Vy(v)
        for _ in range(M - 1):
            w = self.apply_Vy(self.apply_Vx(w))
        return w

    def strip_amplitude(self, M: int) -> complex:
        """<+^n| V_y (V_x V_y)^{M-1} |+^n>  (the quantum-circuit observable)."""
        N = 2 ** self.n
        v = np.full(N, 1.0 / np.sqrt(N), dtype=complex)
        return np.vdot(v, self.strip_vector(M))        # <+|w>, |+> real

    def Z_strip(self, M: int) -> complex:
        pref = (2.0 * sinh2(self.K1)) ** (self.n * (M - 1) / 2.0) * 2.0 ** self.n
        return pref * self.strip_amplitude(M)

    def Z_torus(self, M: int) -> complex:
        """(2 sinh 2K1)^{nM/2} Tr[(V_x V_y)^M] by applying T^M to all basis vectors (dense; n <= ~14)."""
        N = 2 ** self.n
        if N > 2 ** 14:
            raise ValueError("dense trace limited to n <= 14; use Z_torus_sampled")
        tr = 0.0 + 0.0j
        for b in range(N):
            v = np.zeros(N, dtype=complex); v[b] = 1.0
            for _ in range(M):
                v = self.apply_T(v)
            tr += v[b]
        return (2.0 * sinh2(self.K1)) ** (self.n * M / 2.0) * tr

    def Z_torus_sampled(self, M: int, nsamp: int = 64, rng=None) -> complex:
        """Trace by random-phase vectors (typicality); error ~ 1/sqrt(nsamp)."""
        rng = np.random.default_rng(rng)
        N = 2 ** self.n
        acc = 0.0 + 0.0j
        for _ in range(nsamp):
            v = np.exp(2j * np.pi * rng.random(N)) / np.sqrt(N)
            w = v.copy()
            for _ in range(M):
                w = self.apply_T(w)
            acc += np.vdot(v, w) * N
        return (2.0 * sinh2(self.K1)) ** (self.n * M / 2.0) * acc / nsamp


# ----------------------------------------------------------------------------- the predicted zero locus
def onsager_kaufman_min(K1: complex, K2: complex, ngrid: int = 181) -> float:
    """min over real momenta (t1,t2) of |c1 c2 - s1 cos t1 - s2 cos t2|.

    In the thermodynamic limit the Fisher zeros of the anisotropic model condense on the set of
    complex couplings where this minimum is 0 (the Onsager-Kaufman factor vanishes for some real
    momenta; for K1 = K2 this is the circle |sinh 2K| = 1).  Returns the minimum, so that the locus
    is its zero set.
    """
    c1, c2 = np.cosh(2 * K1), np.cosh(2 * K2)
    s1, s2 = np.sinh(2 * K1), np.sinh(2 * K2)
    t = np.linspace(-np.pi, np.pi, ngrid)
    f = c1 * c2 - s1 * np.cos(t)[:, None] - s2 * np.cos(t)[None, :]
    return float(np.min(np.abs(f)))


def circuit_couplings(theta: float, phi: float) -> tuple[complex, complex]:
    """Couplings realised by the unitary kicked-Ising circuit with RZZ angle theta and RX angle phi:
    K2 = i theta (row coupling), K1* = i phi (dual column coupling), K1 = -1/2 log(tanh(i phi))."""
    K2 = 1j * theta
    K1 = primal_coupling(1j * phi)
    return K1, K2


# ----------------------------------------------------------------------------- zero finding
def find_zeros_on_grid(fun, re_range, im_range, nre=61, nim=61, refine=True, tol=1e-10):
    """Locate zeros of a holomorphic function fun(z) in a rectangle: sign changes of Re and Im on a grid
    (candidate cells), then Newton refinement with numerical derivative.  Returns a list of complex zeros."""
    xs = np.linspace(*re_range, nre); ys = np.linspace(*im_range, nim)
    Zg = np.array([[fun(x + 1j * y) for x in xs] for y in ys])
    zeros = []
    sr, si = np.sign(Zg.real), np.sign(Zg.imag)
    for i in range(nim - 1):
        for j in range(nre - 1):
            cr = sr[i:i+2, j:j+2]; ci = si[i:i+2, j:j+2]
            if cr.min() != cr.max() and ci.min() != ci.max():
                z = (xs[j] + xs[j+1]) / 2 + 1j * (ys[i] + ys[i+1]) / 2
                if refine:
                    h = 1e-6
                    for _ in range(50):
                        f = fun(z)
                        if abs(f) < tol:
                            break
                        d = (fun(z + h) - fun(z - h)) / (2 * h)
                        if d == 0:
                            break
                        z = z - f / d
                    if abs(fun(z)) < 1e-6 * max(1.0, abs(Zg).max()):
                        zeros.append(z)
                else:
                    zeros.append(z)
    # dedupe
    out = []
    for z in zeros:
        if all(abs(z - w) > 1e-6 for w in out):
            out.append(z)
    return out
