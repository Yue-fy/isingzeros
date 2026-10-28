"""
Itemised Euro-Q-Exa request from results/budget.csv, including per-circuit time and LRZ's accounting unit.

    python -m isingzeros.request [--shot-ms 1.0] [--circuit-s 2.5] [--twirls 4] [--contingency 0.2]

Device (QPU wall-clock) time = shots x shot time + circuits x per-circuit time.
  - per-circuit time: LRZ reports a median of 2.5 s per circuit on Q-Exa (Friendly User Pilot Phase); it includes the
    shots of a typical circuit, so charging it in full on top of our shot time is an upper bound;
  - every measurement point is run as `twirls` randomly Pauli-twirled circuit instances (shots split between them);
  - noise amplification (gate folding x2, x3): the WP2 6x3 scan (40 angles x 2.5e4 shots per level) and the smallest
    circuit of each return-probability campaign; repeats: 10 % of the base points and shots; calibration: readout
    confusion circuits and drift monitors, 30 sessions x 3 circuits, 1e6 shots.
LRZ accounting unit (https://doku.lrz.de/definition-of-the-computing-unit-for-the-lrz-quantum-systems-1898715012.html):
  equivalent qubit-seconds  eqs = t x n(S) x T1qg(Q-Exa)/T1qg(S);  here n = 53 (the call lists Euro-Q-Exa as a 53-qubit lattice) and gate-time ratio 1 (to be confirmed).
Writes results/request.json.
"""
from __future__ import annotations
import argparse, json, math
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--budget', default='results/budget.csv'); ap.add_argument('--out', default='results/request.json')
    ap.add_argument('--shot-ms', type=float, default=1.0); ap.add_argument('--circuit-s', type=float, default=2.5)
    ap.add_argument('--twirls', type=int, default=4); ap.add_argument('--contingency', type=float, default=0.2)
    ap.add_argument('--n-qubits', type=int, default=53); ap.add_argument('--gate-ratio', type=float, default=1.0)
    a = ap.parse_args()
    b = pd.read_csv(a.budget); b = b[b.system == 'Euro-Q-Exa']
    cond = (b.campaign == 'C_zero2D_conditional') | ((b.campaign == 'B_slab') & (b.qubits == 32))
    base = b[~cond]
    # noise amplification: 6x3 scan at 2.5e4 shots per angle, plus the smallest circuit of each return-probability campaign
    fr63 = base[(base.campaign == 'C_FR') & (base.L == 6)]
    na_pts, na_shots = 2 * int(fr63.points.sum()), 2 * int(fr63.points.sum()) * 25000
    for c in ('A_chain', 'A2_aspect', 'B_slab'):
        g = base[base.campaign == c]; sm = g[g.qubits == g.qubits.min()]
        na_pts += 2 * int(sm.points.sum()); na_shots += 2 * int(sm.shots_total.sum())
    rep_pts, rep_shots = round(0.1 * base.points.sum()), round(0.1 * base.shots_total.sum())
    cal_circ, cal_shots = 30 * 3, 1_000_000
    items = [('Base plan', int(base.points.sum()) * a.twirls, int(base.shots_total.sum())),
             ('Noise amplification', na_pts * a.twirls, na_shots),
             ('Calibration', cal_circ, cal_shots),
             ('Repeats', rep_pts * a.twirls, rep_shots)]
    rows = []
    for name, circ, shots in items:
        rows.append(dict(item=name, circuits=circ, shots=shots, shot_h=shots * a.shot_ms / 1000 / 3600))
    shots_h = sum(r['shot_h'] for r in rows); circ = sum(r['circuits'] for r in rows)
    over_h = circ * a.circuit_s / 3600
    sub = shots_h + over_h; cont = a.contingency * sub; total = sub + cont
    req_h = math.ceil(total)
    eqs = lambda h: h * 3600 * a.n_qubits * a.gate_ratio
    # core: WP1 rings up to 12 qubits + WP2 (scans and sign test) with their noise amplification
    core = b[b.campaign.isin(['C_FR', 'C_sign']) | ((b.campaign == 'A_chain') & (b.qubits <= 12))]
    ring_small = core[(core.campaign == 'A_chain') & (core.qubits == 8)]
    core_pts = int(core.points.sum()) + 2 * int(fr63.points.sum()) + 2 * int(ring_small.points.sum())
    core_shots = int(core.shots_total.sum()) + 2 * int(fr63.points.sum()) * 25000 + 2 * int(ring_small.shots_total.sum())
    core_h = (core_shots * a.shot_ms / 1000 + (core_pts * a.twirls + cal_circ) * a.circuit_s) / 3600 + cal_shots * a.shot_ms / 1000 / 3600
    out = dict(params=vars(a), items=rows, shots_total=sum(r['shots'] for r in rows), shot_hours=shots_h, circuits=circ,
               overhead_hours=over_h, subtotal_hours=sub, contingency_hours=cont, total_hours=total, request_hours=req_h,
               request_eqs=eqs(req_h), core_hours=core_h, core_eqs=eqs(core_h))
    json.dump(out, open(a.out, 'w'), indent=1)
    for r in rows:
        print(f"{r['item']:22s} circuits {r['circuits']:6d}  shots {r['shots']:.3g}  shot time {r['shot_h']:.2f} h")
    print(f"shots {out['shots_total']:.3g} -> {shots_h:.2f} h;  circuits {circ} x {a.circuit_s} s -> {over_h:.2f} h;  "
          f"subtotal {sub:.2f} h;  contingency {cont:.2f} h;  total {total:.2f} h -> request {req_h} h = {eqs(req_h):,.0f} eqs")
    print(f"core (WP1 <= 12 qubits + WP2): {core_h:.2f} h = {eqs(core_h):,.0f} eqs")


if __name__ == '__main__':
    main()
