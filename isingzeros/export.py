"""
Hardware export.  Produces (i) OpenQASM 3 files for Euro-Q-Exa (LRZ accepts abstract circuits and
transpiles on site), (ii) circuits transpiled to IQM native gates {r, cz} with a user-supplied
coupling map (VLQ: star topology, see docs.it4i.cz; Euro-Q-Exa lattice), (iii) a JSON job list for
Piast-Q (AQT ARNICA: native gates RZ, R, RXX, angles in units of pi, one terminal measurement),
and (iv) an IBM Qiskit Runtime submission function (Open Plan: SamplerV2, error mitigation via
twirling).  Credentials are never stored here; read them from environment variables.

    python -m isingzeros.export qasm   --L 8 --M 4 --theta 0.3:1.5:25 --phi 0.6 --outdir jobs/eqexa
    python -m isingzeros.export ibm    --L 8 --M 4 --theta 0.3:1.5:13 --phi 0.6 --shots 20000 --backend ibm_torino
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
from qiskit import transpile, qasm3
from .circuits import kicked_ising_unitary


def _parse_range(s):
    a, b, n = s.split(':'); return np.linspace(float(a), float(b), int(n))


def export_qasm(a):
    os.makedirs(a.outdir, exist_ok=True)
    manifest = []
    for i, th in enumerate(_parse_range(a.theta)):
        qc = kicked_ising_unitary(a.L, a.M, th, a.phi, W=a.W, theta3=a.theta3)
        # Euro-Q-Exa: abstract circuit is fine; we still pre-transpile to {r/rz,sx,x,cz} to control depth
        t = transpile(qc, basis_gates=['rz', 'sx', 'x', 'cz'], optimization_level=3, seed_transpiler=1)
        fn = os.path.join(a.outdir, f'kicked_ising_L{a.L}_W{a.W}_M{a.M}_phi{a.phi:.3f}_theta{th:.4f}.qasm')
        with open(fn, 'w') as f:
            qasm3.dump(t, f)
        manifest.append({'file': os.path.basename(fn), 'theta': float(th), 'phi': a.phi, 'L': a.L, 'W': a.W, 'M': a.M,
                         'two_qubit': dict(t.count_ops()).get('cz', 0), 'depth': t.depth(), 'shots': a.shots})
    json.dump(manifest, open(os.path.join(a.outdir, 'manifest.json'), 'w'), indent=1)
    print(f'{len(manifest)} circuits -> {a.outdir}')


def piast_native_list(qc):
    """Piast-Q / AQT ARNICA expects native gates RZ(theta), R(theta, phi), RXX(theta) with angles in
    units of pi and a single final measurement.  We transpile to {rz, r, rxx} and emit the list."""
    t = transpile(qc, basis_gates=['rz', 'r', 'rxx'], optimization_level=3, seed_transpiler=1)
    ops = []
    for inst in t.data:
        name = inst.operation.name
        q = [t.find_bit(b).index for b in inst.qubits]
        if name == 'measure':
            continue
        params = [float(p) / np.pi for p in inst.operation.params]
        ops.append({'operation': name.upper(), 'qubits': q, 'params_in_pi': params})
    ops.append({'operation': 'MEASURE'})
    return ops, t


def export_piast(a):
    os.makedirs(a.outdir, exist_ok=True)
    jobs = []
    for th in _parse_range(a.theta):
        ops, t = piast_native_list(kicked_ising_unitary(a.L, a.M, th, a.phi, W=a.W, theta3=a.theta3))
        jobs.append({'theta': float(th), 'phi': a.phi, 'n_qubits': a.L * a.W, 'shots_per_job': 200,
                     'jobs_needed': int(np.ceil(a.shots / 200)), 'rxx_count': dict(t.count_ops()).get('rxx', 0),
                     'circuit': ops})
    json.dump(jobs, open(os.path.join(a.outdir, 'piastq_jobs.json'), 'w'))
    print(f'{len(jobs)} Piast-Q job specs -> {a.outdir}/piastq_jobs.json  (200 shots/job, <=4 concurrent connections)')


def run_ibm(a):
    """IBM Quantum Open Plan.  Requires: pip install qiskit-ibm-runtime; token in env QISKIT_IBM_TOKEN
    (QiskitRuntimeService.save_account once).  Submits one SamplerV2 job with all theta circuits."""
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
    service = QiskitRuntimeService()
    backend = service.backend(a.backend) if a.backend else service.least_busy(operational=True, simulator=False, min_num_qubits=a.L * a.W)
    circs = []
    thetas = _parse_range(a.theta)
    for th in thetas:
        qc = kicked_ising_unitary(a.L, a.M, th, a.phi, W=a.W, theta3=a.theta3)
        circs.append(transpile(qc, backend=backend, optimization_level=3, seed_transpiler=1))
    sampler = Sampler(mode=backend)
    sampler.options.twirling.enable_gates = True          # Pauli twirling of two-qubit gates
    sampler.options.twirling.enable_measure = True        # readout twirling (TREX-like)
    sampler.options.dynamical_decoupling.enable = True
    job = sampler.run(circs, shots=a.shots)
    print('job id', job.job_id(), 'backend', backend.name)
    res = job.result()
    out = []
    n = a.L * a.W
    for th, r in zip(thetas, res):
        counts = r.data.c.get_counts()
        tot = sum(counts.values())
        out.append({'theta': float(th), 'phi': a.phi, 'return_prob': counts.get('0' * n, 0) / tot, 'shots': tot})
    os.makedirs(a.outdir, exist_ok=True)
    json.dump({'backend': backend.name, 'job_id': job.job_id(), 'L': a.L, 'W': a.W, 'M': a.M, 'rows': out},
              open(os.path.join(a.outdir, f'ibm_{backend.name}_L{a.L}M{a.M}.json'), 'w'), indent=1)
    print(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['qasm', 'piast', 'ibm'])
    ap.add_argument('--L', type=int, default=8); ap.add_argument('--M', type=int, default=4); ap.add_argument('--W', type=int, default=1)
    ap.add_argument('--theta3', type=float, default=0.0); ap.add_argument('--phi', type=float, default=0.6)
    ap.add_argument('--theta', type=str, default='0.3:1.5:13'); ap.add_argument('--shots', type=int, default=20000)
    ap.add_argument('--backend', type=str, default=''); ap.add_argument('--outdir', type=str, default='jobs')
    a = ap.parse_args()
    {'qasm': export_qasm, 'piast': export_piast, 'ibm': run_ibm}[a.mode](a)


if __name__ == '__main__':
    main()
