"""
Figures of the ISINGZEROS proposal (EuroHPC Quantum Access Pilot, October 2026) and of its feasibility proof.

    python -m isingzeros.propfigs fig1             # Ising strip -> kicked-Ising circuit -> zeros of the 6x3 strip
    python -m isingzeros.propfigs fig2 [--nproc 8] # feasibility of the reconstruction (needs results/verify/v5, v6)
    python -m isingzeros.propfigs feas             # figure of the feasibility proof (needs results/emulator.json)

Both figures are drawn at their final printed size (17 cm wide, 8 pt text) and saved without bbox cropping,
so the document must insert them at width = 17 cm.  Outputs figs/prop_fig{1,2}.{pdf,png,svg} and the source
data figs/prop_fig{1,2}_data.json.  If figs/prop_fig2_data.json exists, fig2 reuses it; otherwise it recomputes
the noisy return probabilities and the bootstrap clouds (density matrix, a few minutes with --nproc 8).
Noise model of Fig. 2: depolarising p2 after every CZ, p2/10 after every physical one-qubit gate, rz error-free,
1 % symmetric readout error; angle-independent compilation (circuits.compile_fixed).
Fonts: Liberation Sans (Arial metrics) if found in $ISINGZEROS_FONTDIR, else DejaVu Sans.
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Rectangle, Arc
from . import fejer as fr

for _f in glob.glob(os.path.join(os.environ['ISINGZEROS_FONTDIR'], '*.ttf')) if os.environ.get('ISINGZEROS_FONTDIR') else []:
    font_manager.fontManager.addfont(_f)
FS = 8.0
mpl.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
    'pdf.fonttype': 42, 'svg.fonttype': 'none', 'font.size': FS, 'axes.labelsize': FS, 'xtick.labelsize': FS - 0.5,
    'ytick.labelsize': FS - 0.5, 'legend.fontsize': FS - 0.5, 'axes.spines.right': False, 'axes.spines.top': False,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.minor.width': 0.4,
    'ytick.minor.width': 0.4, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.minor.size': 1.5,
    'ytick.minor.size': 1.5, 'legend.frameon': False, 'mathtext.fontset': 'custom', 'mathtext.rm': 'Liberation Sans',
    'mathtext.it': 'Liberation Sans:italic', 'mathtext.bf': 'Liberation Sans:bold', 'mathtext.fallback': 'cm', 'mathtext.cal': 'Liberation Sans'})
INK, GREY, SIG, ACC, LIGHT = '#222222', '#9a9a9a', '#2c6fbb', '#c8443a', '#dbe6f4'
ZZC, XC = '#e3ecf8', '#fbe4e2'
L, M, PHI = 6, 3, 0.6
W_CM = 17.0
WINDOW = 0.1              # |a0| below which the 6x3 reconstruction is accurate to ~1e-3 (verify V5)
PI_TICKS = ([0, np.pi / 8, np.pi / 4, 3 * np.pi / 8, np.pi / 2], ['0', r'$\pi/8$', r'$\pi/4$', r'$3\pi/8$', r'$\pi/2$'])


def save(fig, name):
    os.makedirs('figs', exist_ok=True)
    for ext in ('pdf', 'png', 'svg'):
        fig.savefig(f'figs/{name}.{ext}', dpi=600 if ext == 'png' else None)
    print(f'-> figs/{name}.pdf/.png/.svg')


def axes_cm(fig, x0, y0, w, h, **kw):
    """Axes placed in centimetres from the lower-left corner of the figure."""
    W, H = fig.get_size_inches() * 2.54
    return fig.add_axes([x0 / W, y0 / H, w / W, h / H], **kw)


def head(fig, x, y, letter, title):
    """Panel letter and short title, positioned in cm."""
    W, H = fig.get_size_inches() * 2.54
    fig.text(x / W, y / H, letter, fontsize=FS + 1, fontweight='bold', va='baseline', ha='left')
    fig.text((x + 0.42) / W, y / H, title, fontsize=FS, va='baseline', ha='left')


# ----------------------------------------------------------------------------- Fig. 1
YQ = np.arange(L)[::-1].astype(float)          # q1 on top, shared by panels a and b
YLIM = (-1.15, 5.75)


def _lattice(ax):
    """The strip drawn as a short cylinder: each row is a ring of L spins (bonds K2), neighbouring rows are
    coupled by K1.  Front half solid, back half faded."""
    ax.set_xlim(-0.75, 3.4); ax.set_ylim(*YLIM); ax.axis('off')
    xs, rx, ry, yc = [0.45, 1.45, 2.45], 0.32, 2.55, 2.45
    ang = np.pi / 2 + 2 * np.pi * np.arange(L) / L
    front = np.sin(ang - np.pi / 2) >= -1e-9                 # k = 0..3 visible, k = 4, 5 behind
    for x0, x1 in zip(xs[:-1], xs[1:]):
        for k in range(L):
            px, py = rx * np.sin(ang[k] - np.pi / 2), ry * np.cos(ang[k] - np.pi / 2)
            ax.plot([x0 + px, x1 + px], [yc + py, yc + py], color=ACC, lw=1.0 if front[k] else 0.7,
                    alpha=1 if front[k] else 0.35, zorder=1 if front[k] else 0)
    for x in xs:
        ax.add_patch(Arc((x, yc), 2 * rx, 2 * ry, theta1=-90, theta2=90, color=SIG, lw=1.5, zorder=1))
        ax.add_patch(Arc((x, yc), 2 * rx, 2 * ry, theta1=90, theta2=270, color=SIG, lw=1.0, alpha=0.35, zorder=0))
        for k in range(L):
            px, py = rx * np.sin(ang[k] - np.pi / 2), ry * np.cos(ang[k] - np.pi / 2)
            ax.scatter([x + px], [yc + py], s=20 if front[k] else 14, color='white', edgecolor=INK if front[k] else GREY,
                       lw=0.8, zorder=3 if front[k] else 0.5)
    ax.text(xs[0] - rx - 0.12, yc, r'$K_2$', color=SIG, ha='right', va='center')
    ax.text((xs[1] + xs[2]) / 2, yc + ry + 0.12, r'$K_1$', color=ACC, ha='center', va='bottom')
    ax.annotate('', xy=(xs[-1] + 0.3, -0.75), xytext=(xs[0] - 0.3, -0.75),
                arrowprops=dict(arrowstyle='-|>', lw=0.7, color=INK, mutation_scale=7))
    ax.text((xs[0] + xs[-1]) / 2, -0.88, 'row to row = circuit time', ha='center', va='top', fontsize=FS - 0.5)


def _circuit(ax):
    ax.set_xlim(0, 10.0); ax.set_ylim(*YLIM); ax.axis('off')
    for k, y in enumerate(YQ):
        ax.plot([0.55, 7.12], [y, y], color=INK, lw=0.6, zorder=0)
        ax.text(0.42, y, rf'$q_{k + 1}$', ha='right', va='center', fontsize=FS - 0.5)

    def box(x, y0, y1, label, fc, w):
        ax.add_patch(FancyBboxPatch((x - w / 2, y0 - 0.3), w, y1 - y0 + 0.6, boxstyle='round,pad=0.02,rounding_size=0.08',
                                    fc=fc, ec=INK, lw=0.55, zorder=2))
        if label:
            ax.text(x, (y0 + y1) / 2, label, ha='center', va='center', fontsize=FS - 0.5, zorder=3)

    def vy(x):
        for i in (0, 2, 4):
            box(x, YQ[i + 1], YQ[i], '', ZZC, 0.40)
        for i in (1, 3):
            box(x + 0.48, YQ[i + 1], YQ[i], '', ZZC, 0.40)
        for y in (YQ[0], YQ[-1]):                             # closing bond q6-q1
            ax.add_patch(Rectangle((x + 0.80, y - 0.28), 0.30, 0.56, fc=ZZC, ec=INK, lw=0.55, zorder=2))
        ax.plot([x + 0.95, x + 0.95], [YQ[-1] + 0.28, YQ[0] - 0.28], color=INK, lw=0.5, ls=(0, (2, 1.5)), zorder=1)
        ax.text(x + 0.48, 5.42, r'$ZZ(\theta)$', ha='center', va='bottom', fontsize=FS - 0.5, color=SIG)

    def vx(x):
        for y in YQ:
            box(x, y, y, '', XC, 0.34)
        ax.text(x, 5.42, r'$X(\varphi)$', ha='center', va='bottom', fontsize=FS - 0.5, color=ACC)

    for y in YQ:
        box(1.0, y, y, 'H', 'white', 0.46)
    vy(1.62); vx(3.20); vy(3.72)
    ax.text(5.25, 2.5, r'$\cdots$', ha='center', va='center', fontsize=FS + 2)
    ax.plot([2.95, 2.95, 4.95, 4.95], [-0.42, -0.58, -0.58, -0.42], color=INK, lw=0.55)
    ax.text(3.95, -0.70, r'repeat $M-1$ times', ha='center', va='top', fontsize=FS - 0.5)
    for y in YQ:
        box(6.15, y, y, 'H', 'white', 0.46)
        ax.add_patch(FancyBboxPatch((6.85 - 0.26, y - 0.28), 0.52, 0.56, boxstyle='round,pad=0.01,rounding_size=0.06',
                                    fc='#eeeeee', ec=INK, lw=0.55, zorder=2))
        ax.add_patch(Arc((6.85, y - 0.12), 0.34, 0.32, theta1=0, theta2=180, lw=0.55, color=INK, zorder=3))
        ax.plot([6.85, 6.97], [y - 0.12, y + 0.12], color=INK, lw=0.55, zorder=3)
    ax.text(7.32, 2.5, r'$P(\theta)$ = fraction' '\nof shots that' '\nreturn all 0', ha='left', va='center', fontsize=FS - 0.5)


def fig1(a):
    zs = fr.exact_zeros(L, M, PHI)
    fig = plt.figure(figsize=(W_CM / 2.54, 5.6 / 2.54))
    axa = axes_cm(fig, 0.0, 0.25, 3.55, 4.55); _lattice(axa)
    axb = axes_cm(fig, 3.55, 0.25, 6.6, 4.55); _circuit(axb)
    head(fig, 0.05, 5.15, 'a', 'Ising strip, $L=6$, $M=3$')
    head(fig, 3.75, 5.15, 'b', 'Kicked-Ising circuit')
    # c: zeros in the complex K2 plane
    ax = axes_cm(fig, 11.75, 0.95, 4.95, 3.95)
    ax.axhspan(-WINDOW, WINDOW, color=LIGHT, lw=0, zorder=0)
    ax.axhline(0, color=SIG, lw=1.4, zorder=1)
    ax.text(0.03, WINDOW + 0.015, 'unitary circuits ($a=0$)', color=SIG, fontsize=FS - 0.5, va='bottom', ha='left')
    ax.scatter([z.imag for z in zs], [-z.real for z in zs], s=15, facecolors='white', edgecolors=GREY, lw=0.8, zorder=2,
               label='mirror image')
    ax.scatter([z.imag for z in zs], [z.real for z in zs], s=15, color=INK, zorder=3, label='zero')
    ax.set_xlim(0, np.pi / 2); ax.set_ylim(-0.66, 0.92)
    ax.set_xticks(*PI_TICKS); ax.set_yticks([-0.5, 0, 0.5])
    ax.set_xlabel(r'$\theta=\mathrm{Im}\,K_2$'); ax.set_ylabel(r'$a=\mathrm{Re}\,K_2$', labelpad=1)
    h, lab = ax.get_legend_handles_labels()
    ax.legend(h[::-1], lab[::-1], loc='upper center', ncol=2, handletextpad=0.1, columnspacing=1.2, borderaxespad=0.15,
              fontsize=FS - 0.5)
    head(fig, 10.75, 5.15, 'c', 'Zeros of the strip ($\\varphi=0.6$)')
    json.dump(dict(L=L, M=M, phi=PHI, zeros=[[z.real, z.imag] for z in zs], window_halfwidth=WINDOW),
              open('figs/prop_fig1_data.json', 'w'), indent=1)
    save(fig, 'prop_fig1')


# ----------------------------------------------------------------------------- Fig. 2 (feasibility)
def _noisy(args):
    t, p2, pro = args
    from .analysis import noisy_joint_probability_dm
    from .circuits import kicked_ising_unitary
    return noisy_joint_probability_dm(kicked_ising_unitary(L, M, t, PHI), p2, p2 / 10, pro)


def _recon_data(a, path='figs/prop_fig2_data.json'):
    if os.path.exists(path) and not a.recompute:
        return json.load(open(path))
    n = fr.degree(L, M); zs = fr.exact_zeros(L, M, PHI)
    th = fr.theta_grid(n, 40); S = 50000; p2, pro = 0.005, 0.01
    if a.nproc > 1:
        from multiprocessing import Pool
        with Pool(a.nproc) as pool:
            Pn = np.array(pool.map(_noisy, [(t, p2, pro) for t in th]))
    else:
        Pn = np.array([_noisy((t, p2, pro)) for t in th])
    Pex = fr.return_probabilities_exact(L, M, PHI, th)
    rng = np.random.default_rng(a.seed)
    k = rng.binomial(S, Pn)
    targets = [z for z in zs if abs(z.real) < 0.05]
    clouds = {}
    for lab, P, floor in (('ideal', Pex, 'none'), ('noisy', Pn, 'onaxis')):
        pts = []
        for _ in range(a.R):
            pb, *_ = fr.sample_and_reconstruct(th, P, S, n, rng, floor, np.pi / 4, symmetric=True)
            pts.append([(m['theta_hat'], m['a_abs_hat']) for m in fr.match(pb, targets)])
        clouds[lab] = np.array(pts)                # [R, target, (theta, |a|)]
    d = dict(L=L, M=M, phi=PHI, theta=th.tolist(), P_exact=Pex.tolist(), P_noisy=Pn.tolist(), counts=k.tolist(), shots=S,
             p2=p2, p_ro=pro, targets=[[z.real, z.imag] for z in targets],
             cloud_ideal=clouds['ideal'].tolist(), cloud_noisy=clouds['noisy'].tolist(), seed=a.seed, R=a.R)
    json.dump(d, open(path, 'w'))
    return d


def fig2(a):
    d = _recon_data(a)
    n = fr.degree(L, M)
    th, Pn, S = np.array(d['theta']), np.array(d['P_noisy']), d['shots']
    Ph = np.array(d['counts']) / S; err = np.sqrt(np.maximum(Ph * (1 - Ph), 1 / S) / S)
    targets = [complex(*z) for z in d['targets']]
    tf = np.linspace(0, np.pi / 2, 721)
    Pf_ex = fr.return_probabilities_exact(L, M, PHI, tf)
    Pf_noisy = fr.trig_eval(fr.trig_fit(th, Pn, n, symmetric=True), tf)   # exact: Pauli noise stays in the cosine space

    fig = plt.figure(figsize=(W_CM / 2.54, 11.2 / 2.54))
    # (a) return probability on one period
    ax = axes_cm(fig, 1.35, 6.55, 6.6, 3.85)
    ax.semilogy(tf, np.maximum(Pf_ex, 1e-7), color=INK, lw=0.9, label='exact')
    ax.semilogy(tf, Pf_noisy, color=SIG, lw=0.9, ls='--', label='with gate noise')
    ax.errorbar(th, Ph, yerr=err, fmt='o', ms=2.3, mfc=SIG, mec=SIG, ecolor=SIG, elinewidth=0.6, capsize=0,
                label='simulated data')
    for z in targets:
        ax.plot([z.imag], [4.5], marker='v', ms=3.5, color=ACC, clip_on=False, zorder=4)
    ax.set_ylim(1e-5, 4.5); ax.set_xlim(0, np.pi / 2)
    ax.set_xticks(*PI_TICKS); ax.set_yticks([1e-4, 1e-2, 1])
    ax.set_xlabel(r'circuit angle $\theta$'); ax.set_ylabel(r'return probability $P(\theta)$')
    ax.legend(loc='lower left', handlelength=1.4, labelspacing=0.25, borderaxespad=0.2, fontsize=FS - 0.5)
    head(fig, 0.15, 10.75, 'a', 'Simulated measurement, 6 × 3 strip, $p_2=0.5\\%$')

    # (b) reconstructed (theta0, |a0|) for the three zeros nearest the axis
    ax = axes_cm(fig, 10.05, 6.55, 6.6, 3.85)
    ci, cn = np.array(d['cloud_ideal']), np.array(d['cloud_noisy'])
    ax.axhspan(-0.01, 0.0, color='white', zorder=0)
    ax.scatter(ci[:, :, 0].ravel(), ci[:, :, 1].ravel(), s=2.0, color=GREY, alpha=0.45, lw=0, rasterized=True,
               label='shot noise only')
    ax.scatter(cn[:, :, 0].ravel(), cn[:, :, 1].ravel(), s=2.0, color=SIG, alpha=0.45, lw=0, rasterized=True,
               label='with gate noise')
    ax.scatter([z.imag for z in targets], [abs(z.real) for z in targets], marker='x', s=24, color=ACC, lw=1.1, zorder=4,
               label='exact zero')
    ax.set_xlim(0.56, 1.01); ax.set_ylim(-0.002, 0.052)
    ax.set_xticks([0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xlabel(r'reconstructed angle $\theta_0$'); ax.set_ylabel(r'distance from axis $|a_0|$')
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], ls='', marker='o', ms=3, color=GREY, label='shot noise only'),
                       Line2D([], [], ls='', marker='o', ms=3, color=SIG, label='with gate noise'),
                       Line2D([], [], ls='', marker='x', ms=4.5, mew=1.1, color=ACC, label='exact zero')],
              loc='upper center', handletextpad=0.2, labelspacing=0.25, borderaxespad=0.1, fontsize=FS - 0.5)
    head(fig, 8.85, 10.75, 'b', 'Zeros recovered from $P(\\theta)$, 200 simulated runs')

    # (c) accuracy against gate error: unitary reconstruction vs direct post-selected grid (verify V6)
    v6 = json.load(open(a.v6))
    rows = sorted([r for r in v6['rows'] if abs(r['a0'] - 0.0213) < 1e-3 and abs(r['lcu_halfwidth'] - 0.03) < 1e-9],
                  key=lambda r: r['p2'])
    p = np.array([r['p2'] for r in rows]) * 100
    ax = axes_cm(fig, 1.35, 1.0, 6.6, 3.85)
    ax.plot([0.075, 1.12], [abs(targets[0].real)] * 2, color=GREY, lw=0.7, ls='--', zorder=0)
    ax.text(0.082, abs(targets[0].real) * 1.12, 'distance of the zero\nfrom the axis', color='#6f6f6f', fontsize=FS - 0.5,
            va='bottom')
    ax.loglog(p, [r['lcu_median_err'] for r in rows], 's-', color=ACC, ms=3.5, lw=1.0)
    ax.loglog(p, [r['fr_median_err'] for r in rows], 'o-', color=SIG, ms=3.5, lw=1.0)
    ax.text(1.22, rows[-1]['lcu_median_err'] * 0.8, 'direct\npost-selection\n' r'$5\times10^6$ shots', color=ACC,
            fontsize=FS - 0.5, ha='left', va='bottom')
    ax.text(1.22, rows[-1]['fr_median_err'], 'this project\n' r'$2\times10^6$ shots', color=SIG, fontsize=FS - 0.5,
            ha='left', va='center')
    ax.set_xlim(0.075, 4.2); ax.set_ylim(8e-4, 0.08)
    ax.set_xticks([0.1, 0.2, 0.5, 1]); ax.set_xticklabels(['0.1', '0.2', '0.5', '1'])
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter()); ax.spines['bottom'].set_bounds(0.075, 1.25)
    ax.xaxis.set_minor_locator(mpl.ticker.FixedLocator([0.08, 0.09, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]))
    ax.set_xlabel(r'two-qubit gate error $p_2$ (%)'); ax.set_ylabel('median error of zero position')
    head(fig, 0.15, 5.2, 'c', 'Accuracy against gate noise')

    # (d) measurability window (shot noise only, verify V5)
    v5 = json.load(open(a.v5))
    rr = [r for r in v5['rows'] if abs(r['a0']) > 1e-6]
    ax = axes_cm(fig, 10.05, 1.0, 5.75, 3.85)
    ax.axvspan(8e-3, WINDOW, color=LIGHT, lw=0, zorder=0)
    norm = mpl.colors.Normalize(vmin=4, vmax=20); cmap = mpl.colormaps['viridis_r']
    sc = ax.scatter([abs(r['a0']) for r in rr], [r['rms_a'] for r in rr], s=5, c=[r['n'] for r in rr], cmap=cmap, norm=norm,
                    lw=0, zorder=2)
    own = [r for r in rr if (r['L'], r['M']) == (L, M) and abs(r['phi'] - PHI) < 1e-9]
    ax.scatter([abs(r['a0']) for r in own], [r['rms_a'] for r in own], s=26, facecolors='none', edgecolors=INK, lw=0.7,
               zorder=3)
    xx = np.logspace(-2.2, 0)
    ax.loglog(xx, xx / 3, color=GREY, lw=0.6, ls='--', zorder=1)
    ax.text(0.0105, 0.0105 / 3 * 1.5, r'error $= |a_0|/3$', color='#6f6f6f', fontsize=FS - 0.5, ha='left', va='bottom',
            rotation=np.degrees(np.arctan((3.85 / np.log10(0.6 / 3e-4)) / (5.75 / np.log10(1.0 / 8e-3)))),
            rotation_mode='anchor')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(8e-3, 1.0); ax.set_ylim(3e-4, 0.6)
    ax.set_xlabel(r'distance from axis $|a_0|$'); ax.set_ylabel(r'error of $|a_0|$')
    cax = axes_cm(fig, 15.95, 1.0, 0.18, 3.85)
    cb = fig.colorbar(sc, cax=cax, ticks=[4, 8, 12, 16, 20]); cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(labelsize=FS - 0.5, length=2, width=0.5); cb.set_label(r'size $n=LM/2$', fontsize=FS - 0.5, labelpad=2)
    head(fig, 8.85, 5.2, 'd', 'Where the method works, nine strip sizes')
    for lab, c in (('ideal', ci), ('noisy', cn)):
        for j, z in enumerate(targets):
            print(f'{lab:5s} zero ({z.real:+.4f},{z.imag:.4f}): |a| = {c[:, j, 1].mean():.4f} +- {c[:, j, 1].std():.4f}, '
                  f'theta = {c[:, j, 0].mean():.4f} +- {c[:, j, 0].std():.4f}')
    save(fig, 'prop_fig2')


# ----------------------------------------------------------------------------- feasibility proof, Fig. 1
def feas(a):
    """Emulator benchmark of the routed circuits (results/emulator.json from isingzeros.emulate): (a) surviving signal
    fraction against CZ count, with the attenuation assumed by the shot budget; (b) node time per noisy trajectory."""
    d = json.load(open(a.emu))
    cases = d['cases']
    fig = plt.figure(figsize=(W_CM / 2.54, 6.4 / 2.54))
    ax = axes_cm(fig, 1.35, 1.0, 8.6, 4.75)
    head(fig, 0.15, 6.05, 'a', 'Signal that survives gate noise')
    cz = np.linspace(0, 410, 200)
    for p2, col, fac, va in ((0.005, SIG, 1.25, 'bottom'), (0.01, ACC, 0.8, 'top')):
        ax.plot(cz, (1 - p2) ** cz, color=col, lw=1.0, zorder=1)
        x0 = 330 if p2 < 0.008 else 290
        ax.text(x0, (1 - p2) ** x0 * fac, f'$p_2$ = {p2 * 100:g} %', color=col, va=va, ha='center', fontsize=FS - 0.5)
        x = np.array([c['cz'] for c in cases]); r = [next(n for n in c['noisy'] if n['p2'] == p2) for c in cases]
        F = np.array([n['F'] for n in r]); e = np.array([n['F_sem'] for n in r])
        ax.errorbar(x, F, yerr=e, fmt='o', ms=3.2, mfc=col, mec='white', mew=0.4, ecolor=col, elinewidth=0.6,
                    capsize=0, zorder=3)
        dm = [(c['cz'], n['F_dm']) for c, n in zip(cases, r) if 'F_dm' in n]
        ax.plot(*zip(*dm), 's', ms=4.2, mfc='none', mec=INK, mew=0.6, zorder=4)
    ax.set_yscale('log'); ax.set_ylim(0.01, 1.3); ax.set_xlim(0, 410)
    ax.spines['bottom'].set_bounds(0, 400); ax.set_xticks([0, 100, 200, 300, 400])
    ax.set_yticks([0.01, 0.03, 0.1, 0.3, 1]); ax.set_yticklabels(['0.01', '0.03', '0.1', '0.3', '1'])
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_xlabel('CZ gates after routing'); ax.set_ylabel('surviving signal fraction $F$')
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=GREY, marker='o', ls='none', ms=3.2, mec='white', mew=0.4),
          Line2D([], [], color=INK, marker='s', ls='none', ms=4.2, mfc='none', mew=0.6),
          Line2D([], [], color=GREY, lw=1.0)]
    ax.legend(hs, ['noisy trajectories', 'density matrix (exact)', 'budget estimate $(1-p_2)^{\\mathrm{CZ}}$'],
              loc='lower left', handlelength=1.6, borderaxespad=0.2)
    bx = axes_cm(fig, 12.2, 1.0, 4.5, 4.75)
    head(fig, 11.0, 6.05, 'b', 'Emulation cost')
    q = np.array([c['qubits'] for c in cases]); t = np.array([np.mean([n['s_per_traj'] for n in c['noisy']]) for c in cases])
    bx.plot(q, t * 1e3, 'o', ms=3.2, color=INK, mec='white', mew=0.4)
    bx.set_yscale('log'); bx.set_xlim(2, 29); bx.set_xticks([4, 8, 12, 16, 20, 24, 28])
    lo, hi = np.floor(np.log10(t.min() * 1e3)), np.ceil(np.log10(t.max() * 1e3))
    bx.set_ylim(10 ** lo, 10 ** hi); bx.set_yticks(10 ** np.arange(lo, hi + 1))
    bx.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
    bx.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    bx.set_xlabel('qubits'); bx.set_ylabel('ms per noisy trajectory')
    save(fig, 'feas_fig1')
    json.dump(dict(source=a.emu, cz=[c['cz'] for c in cases], qubits=q.tolist(), s_per_traj=t.tolist(),
                   F={str(p2): [next(n['F'] for n in c['noisy'] if n['p2'] == p2) for c in cases] for p2 in (0.005, 0.01)}),
              open('figs/feas_fig1_data.json', 'w'), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('fig', choices=['fig1', 'fig2', 'feas', 'all'])
    ap.add_argument('--nproc', type=int, default=1); ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--R', type=int, default=200); ap.add_argument('--recompute', action='store_true')
    ap.add_argument('--v5', default='results/verify/v5_window.json'); ap.add_argument('--v6', default='results/verify/v6_fr_vs_lcu.json')
    ap.add_argument('--emu', default='results/emulator.json')
    a = ap.parse_args()
    for f in (['fig1', 'fig2'] if a.fig == 'all' else [a.fig]):
        globals()[f](a)


if __name__ == '__main__':
    main()
