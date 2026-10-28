"""Figures from the JSON files written by sweeps.py.

    python -m isingzeros.plots exact  results/exact_L8M4.json  results/locus.json  figs/exact_L8M4.png
    python -m isingzeros.plots zeros  results/zeros_L8M4_phi0.6.json              figs/zeros_L8M4.png
    python -m isingzeros.plots noisy  results/noisy_L8M4.json                     figs/noisy_L8M4.png
    python -m isingzeros.plots lcu    results/lcu_L6M3.json                       figs/lcu_L6M3.png
"""
from __future__ import annotations
import json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 150})


def plot_exact(f_exact, f_locus, out):
    d = json.load(open(f_exact)); loc = json.load(open(f_locus))
    th, ph = np.array(d['theta']), np.array(d['phi']); P = np.array(d['return_prob'])
    n = d['L'] * d['W']
    lam = -np.log(np.maximum(P, 1e-300)) / (2 * n)          # rate function per site
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.1))
    im = ax[0].pcolormesh(ph, th, lam, shading='auto', cmap='viridis')
    fig.colorbar(im, ax=ax[0], label=r'$\lambda=-\log P/(2n)$')
    ax[0].set_xlabel(r'$\phi$ (RX angle, $K_1^*=i\phi$)'); ax[0].set_ylabel(r'$\theta$ (RZZ angle, $K_2=i\theta$)')
    ax[0].set_title(f'return probability, L={d["L"]}, W={d["W"]}, M={d["M"]}')
    lth, lph, ok = np.array(loc['theta']), np.array(loc['phi']), np.array(loc['ok_min'])
    ax[0].contour(lph, lth, ok, levels=[0.02], colors='w', linewidths=0.8)
    ax[1].pcolormesh(lph, lth, ok, shading='auto', cmap='magma')
    ax[1].contour(lph, lth, ok, levels=[0.02], colors='w', linewidths=0.8)
    ax[1].set_xlabel(r'$\phi$'); ax[1].set_title('predicted zero locus (Onsager–Kaufman factor = 0, white)')
    fig.tight_layout(); fig.savefig(out); print('->', out)


def plot_zeros(f, out):
    d = json.load(open(f))
    z = np.array(d['zeros']) if d['zeros'] else np.zeros((0, 2))
    thr, thi, ok = np.array(d['locus_theta_r']), np.array(d['locus_theta_i']), np.array(d['ok_min'])
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    ax.pcolormesh(thr, thi, np.log10(ok + 1e-6), shading='auto', cmap='Greys_r', alpha=0.6)
    ax.contour(thr, thi, ok, levels=[0.02], colors='C1', linewidths=1.0)
    if len(z):
        ax.plot(z[:, 0], z[:, 1], 'o', ms=4, color='C0', label=f'zeros of $Z_{{strip}}$, L={d["L"]}, M={d["M"]}')
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xlabel(r'Re $\theta$  (= Im $K_2$, measured on hardware)'); ax.set_ylabel(r'Im $\theta$  (= $-$Re $K_2$)')
    ax.set_title(f'complex-$\\theta$ plane at $\\phi$={d["phi"]}; orange: thermodynamic locus')
    ax.legend(loc='upper right', fontsize=7); fig.tight_layout(); fig.savefig(out); print('->', out)


def plot_noisy(f, out):
    d = json.load(open(f)); r = d['rows']
    th = [x['theta'] for x in r]
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    ax.plot(th, [x['exact'] for x in r], 'k-', lw=1, label='transfer matrix (exact)')
    ax.plot(th, [x['ideal_shots'] for x in r], 'o', ms=3, color='C0', label=f'ideal, {d["shots"]} shots')
    ax.plot(th, [x['noisy'] for x in r], 's', ms=3, color='C3', label=f'noisy, $p_2$={d["p2"]}, $p_{{ro}}$={d["p_ro"]}')
    ax.set_xlabel(r'$\theta$'); ax.set_ylabel(r'$P=|\langle +|U|+\rangle|^2$'); ax.set_yscale('log')
    ax.set_title(f'L={d["L"]}, W={d["W"]}, M={d["M"]}, $\\phi$={d["phi"]}; {d["two_qubit_gates"]} CZ, depth {d["depth"]}', fontsize=8)
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(out); print('->', out)


def plot_lcu(f, out):
    d = json.load(open(f)); r = d['rows']
    a = [x['a_zz'] for x in r]
    fig, ax = plt.subplots(1, 2, figsize=(7, 3))
    ax[0].plot(a, [x['exact'] for x in r], 'k-', label='exact (complex $K_2$)')
    ax[0].plot(a, [x['estimate'] for x in r], 'o', color='C0', label=f'LCU post-selected, {d["shots"]} shots')
    ax[0].set_xlabel(r'Re $K_2$ = $a$'); ax[0].set_ylabel(r'$|\langle +|U(K)|+\rangle|^2$'); ax[0].legend(fontsize=7)
    ax[1].plot(a, [x['p_success'] for x in r], 's', color='C2'); ax[1].set_xlabel(r'$a$'); ax[1].set_ylabel('post-selection success')
    ax[1].set_yscale('log')
    fig.suptitle(f'off-axis couplings: L={d["L"]}, M={d["M"]}, $\\theta$={d["theta"]}, $\\phi$={d["phi"]}', fontsize=9)
    fig.tight_layout(); fig.savefig(out); print('->', out)


def plot_localisation(f, out):
    """2D zero localisation (analysis.demo): joint counts on the local (a, theta) grid, exact zero,
    fitted zero with the parametric-bootstrap cloud."""
    d = json.load(open(f))
    pts = np.array(d['grid']); k = np.array(d['counts'], float); S = d['shots']
    a_vals = np.unique(np.round(pts[:, 0], 6)); t_vals = np.unique(np.round(pts[:, 1], 6))
    grid = np.full((len(t_vals), len(a_vals)), np.nan)
    for (aa, tt), kk in zip(pts, k):
        grid[np.argmin(abs(t_vals - tt)), np.argmin(abs(a_vals - aa))] = kk
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.1))
    da = a_vals[1] - a_vals[0]; dt = t_vals[1] - t_vals[0]
    im = ax[0].imshow(grid, origin='lower', aspect='auto', cmap='viridis',
                      extent=[a_vals[0] - da / 2, a_vals[-1] + da / 2, t_vals[0] - dt / 2, t_vals[-1] + dt / 2])
    plt.colorbar(im, ax=ax[0], label=f'joint events ({S} shots/point)')
    for axx in ax:
        axx.plot(d['exact_zero'][0], d['exact_zero'][1], 'r+', ms=12, mew=2, label='exact finite-size zero')
        axx.plot(d['fit']['a0'], d['fit']['theta0'], 'wx' if axx is ax[0] else 'kx', ms=9, mew=2, label='MLE fit')
        axx.set_xlabel(r'$a=\mathrm{Re}\,K_2$'); axx.set_ylabel(r'$\theta=\mathrm{Im}\,K_2$')
    ax[1].plot(d['bootstrap_a'], d['bootstrap_theta'], '.', color='C0', ms=3, alpha=0.5, label='parametric bootstrap')
    ax[1].set_xlim(ax[0].get_xlim()); ax[1].set_ylim(ax[0].get_ylim())
    ax[1].legend(fontsize=7, loc='upper right')
    noise = f'circuit-level noise, $p_2$={d["aer_p2"]}' if d.get('aer_p2', 0) > 0 else f'signal x{d["signal"]}, floor {d["floor"]:.0e}'
    fig.suptitle(f'2D zero localisation, L={d["L"]}, M={d["M"]}, $\\phi$={d["phi"]} ({noise}): '
                 f'error {d["error"]:.4f}, 95% CI widths {d["ci_a"][1]-d["ci_a"][0]:.4f} / {d["ci_theta"][1]-d["ci_theta"][0]:.4f}', fontsize=8)
    fig.tight_layout(); fig.savefig(out); print('->', out)


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'localisation':
        plot_localisation(sys.argv[2], sys.argv[3]); sys.exit()
    if mode == 'exact':
        plot_exact(sys.argv[2], sys.argv[3], sys.argv[4])
    elif mode == 'zeros':
        plot_zeros(sys.argv[2], sys.argv[3])
    elif mode == 'noisy':
        plot_noisy(sys.argv[2], sys.argv[3])
    elif mode == 'lcu':
        plot_lcu(sys.argv[2], sys.argv[3])
