"""Reproduce geometry plots, objective cost and static QP checks (no ROS)."""
import contextlib
import io
import json
from pathlib import Path
import runpy
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2] / 'multi_purpose_mpc_ros'
sys.path.insert(0, str(ROOT))
from test.probe_support import configured_mpc
from multi_purpose_mpc_ros.core.precomputed_lane_reference import PrecomputedLaneReference, design_parameters, apply_precomputed_heading


def main():
    ns = runpy.run_path(str(ROOT/'scripts/generate_l0_reference.py'))
    design = ns['LaneDesign']()
    csv = ROOT / 'env/centerline/l0_reference_wp220_290.csv'
    rows = np.genfromtxt(csv, delimiter=',', names=True)
    _, _, metrics = design.validate(rows['e_y'])
    q, _, k, steering, _ = design.geometry(rows['e_y'])
    old, _, oldk, oldsteer, _ = design.geometry(design.base[design.ids])
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    axes[0].plot(*old.T, '--', label='Original L0 midpoint')
    axes[0].plot(*q.T, label='Precomputed L0 reference')
    axes[0].plot(*design.p[design.ids].T, alpha=.5, label='Center')
    for a, _, _ in design.walls[:1] + design.walls[2:3]:
        axes[0].plot(*a.T, 'k-', linewidth=1)
    for i in range(230,281,5):
        axes[0].annotate(str(i), design.p[i], fontsize=8)
    axes[0].set_aspect('equal')
    axes[0].set(xlabel='map x relative to WP260 [m]', ylabel='map y [m]', title='Physical walls preserved')
    axes[0].legend()
    axes[1].plot(design.ids[1:-1], np.rad2deg(oldsteer), '--', label='Original L0')
    axes[1].plot(design.ids[1:-1], np.rad2deg(steering), label='Precomputed L0')
    axes[1].axhline(18,color='red',linestyle=':');axes[1].axhline(-18,color='red',linestyle=':')
    axes[1].axvspan(230,255,alpha=.1,color='red',label='Existing L0 entry prohibition')
    axes[1].set(xlabel='Center WP',ylabel='Required steer at 35 km/h [deg]',title='Includes configured understeer')
    axes[1].legend();axes[1].grid()
    fig.tight_layout();fig.savefig(Path(__file__).with_name('comparison.png'),dpi=160)
    mpc = configured_mpc()
    path = mpc.model.reference_path
    load_start = time.perf_counter()
    profile = PrecomputedLaneReference.load(csv, path, ROOT, design_parameters(design.cfg),
                    heading_gain=design.cfg['reference_path']['l0_precomputed_heading_gain'])
    load_ms = (time.perf_counter()-load_start)*1000
    timing = {}
    solves = {}
    path.target_lane_idx=0;path.is_overtaking=True
    for enabled in (False, True):
        path.precomputed_lane_reference = profile if enabled else None
        samples=[]
        for _ in range(1000):
            start=time.perf_counter_ns()
            lateral=np.array([mpc._compute_lane_center(250+n,0) for n in range(mpc.N+1)])
            headings=np.zeros(mpc.N+1)
            apply_precomputed_heading(path,250,lateral,headings,0,None,False,None)
            samples.append((time.perf_counter_ns()-start)/1e6)
        timing[str(enabled)]=dict(median_ms=float(np.median(samples)),p95_ms=float(np.percentile(samples,95)))
        results=[]
        for wp_id in range(220,291):
            wp=path.get_waypoint(wp_id);row=rows[wp_id-220]
            mpc.model.update_states(wp.x-row['e_y']*np.sin(wp.psi),wp.y+row['e_y']*np.cos(wp.psi),wp.psi+row['e_psi'])
            mpc.model.wp_id=wp_id
            mpc.model.spatial_state=mpc.model.t2s(wp,mpc.model.temporal_state)
            with contextlib.redirect_stdout(io.StringIO()):
                mpc._init_problem(mpc.N,0.)
                result=mpc.optimizer.solve()
            results.append(dict(wp=wp_id,status=result.info.status,iterations=result.info.iter,
                                solve_ms=result.info.solve_time*1000))
        solves[str(enabled)]=results
    report=dict(geometry=metrics,load_ms=load_ms,reference_horizon_timing=timing,
                static_qp_checks=solves,
                scope='Static poses on candidate, original MPC bounds/settings, no traffic, no AWSIM or closed-loop validation')
    Path(__file__).with_name('verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='static_qp_checks'},indent=2))
    for enabled,results in solves.items():
        print(enabled,{s:sum(r['status']==s for r in results) for s in set(r['status'] for r in results)},
              'max_iter',max(r['iterations'] for r in results))


if __name__=='__main__':
    main()
