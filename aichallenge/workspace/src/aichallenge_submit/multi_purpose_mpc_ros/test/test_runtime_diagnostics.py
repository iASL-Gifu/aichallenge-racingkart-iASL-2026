import json
from types import SimpleNamespace

import pytest

from multi_purpose_mpc_ros.core import runtime_diagnostics as module
from multi_purpose_mpc_ros.core.MPC import MPC


def test_clock_wait_overrun_and_reset(monkeypatch):
    clock = SimpleNamespace(wall=0., thread=0., process=0.)
    monkeypatch.setattr(module, 'time', SimpleNamespace(
        perf_counter=lambda: clock.wall, thread_time=lambda: clock.thread,
        process_time=lambda: clock.process))
    diag = module.RuntimeDiagnostics(.025, 1.)
    diag.begin(10.)
    diag.wait_ms = 10.
    clock.wall = .05; clock.thread = .02; clock.process = .03
    diag.finish(True)
    assert diag.samples['work_wall_ms'][0] == pytest.approx(40.)
    assert diag.samples['non_thread_wall_ms'][0] == pytest.approx(20.)
    assert diag.counts['work_over_period'] == 1
    assert diag.report({}) is None
    clock.wall = 1.
    row = json.loads(diag.report({}).split(' ', 1)[1])
    assert row['counts']['fallback_owned_cycles'] == 1
    assert row['process_cpu_pct'] == 3.
    assert not diag.counts and not diag.samples
    diag.begin(9.)
    assert diag.counts['sim_clock_backwards'] == 1


@pytest.mark.parametrize('fail', [False, True])
def test_mpc_wrapper_preserves_result_exception_and_records_failure(fail):
    mpc = MPC.__new__(MPC)
    diag = module.RuntimeDiagnostics(.025)
    diag.begin(0.)
    mpc._runtime_diagnostics = diag
    mpc._runtime_role = lambda: 'probe_mpc'
    result = object()
    def implementation():
        mpc.last_solution_status = 'solved'
        mpc.failure_reason = None
        if fail:
            raise RuntimeError('original error')
        return result
    mpc._get_control_impl = implementation
    if fail:
        with pytest.raises(RuntimeError, match='original error'):
            mpc.get_control()
    else:
        assert mpc.get_control() is result
    assert diag.counts['probe_mpc_calls'] == 1
    assert diag.counts['probe_mpc_exception'] == int(fail)
    assert len(diag.samples['probe_mpc_wall_ms']) == 1


def test_solver_status_is_logged_even_when_unsolved():
    mpc = MPC.__new__(MPC)
    diag = module.RuntimeDiagnostics(.025)
    diag.begin(0.)
    mpc._runtime_diagnostics = diag
    mpc._runtime_role = lambda: 'live_mpc'
    result = SimpleNamespace(info=SimpleNamespace(status='primal infeasible', iter=75))
    mpc.optimizer = SimpleNamespace(solve=lambda: result)
    mpc._get_control_impl = lambda: mpc._solve_with_runtime_timing()
    assert mpc.get_control() is result
    assert diag.counts['live_mpc_solver:primal infeasible'] == 1
    assert diag.samples['live_mpc_iterations'][0] == 75


def test_identity_uses_file_contents_and_handles_missing(tmp_path):
    path = tmp_path/'config.yaml'
    path.write_text('a: 1\n')
    first = json.loads(module.runtime_identity({'config': path, 'missing': tmp_path/'none'}).split(' ', 1)[1])
    path.write_text('a: 2\n')
    second = json.loads(module.runtime_identity({'config': path}).split(' ', 1)[1])
    assert first['files']['config']['sha256'] != second['files']['config']['sha256']
    assert 'error' in first['files']['missing']


@pytest.mark.parametrize('fail', [False, True])
def test_whole_control_records_early_return_and_exception(fail):
    from .test_overtake_session import controller_method
    diag = module.RuntimeDiagnostics(.025)
    result = object()
    def control():
        if fail:
            raise RuntimeError('control failed')
        return result
    mpc = object()
    controller = SimpleNamespace(
        _runtime_diagnostics=diag, _control=control,
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=0)),
        _car=SimpleNamespace(wp_id=260, reference_path=SimpleNamespace(target_lane_idx=0)),
        _mpc=mpc, _mpcN_center=mpc,
        _reference_pathN_center=SimpleNamespace(), _steering_fallback_armed=True)
    run = controller_method('_run_control_with_diagnostics')
    if fail:
        with pytest.raises(RuntimeError, match='control failed'):
            run(controller)
    else:
        assert run(controller) is result
    assert diag.counts['cycles'] == 1
    assert diag.counts['fallback_owned_cycles'] == 1
    assert not diag.active


def test_stage_measurement_excludes_wait_and_covers_early_return(monkeypatch):
    clock = SimpleNamespace(wall=0., cpu=0.)
    monkeypatch.setattr(module, 'time', SimpleNamespace(
        perf_counter=lambda: clock.wall, thread_time=lambda: clock.cpu,
        process_time=lambda: clock.cpu))
    diag = module.RuntimeDiagnostics(.025)
    diag.begin(0.)
    clock.wall = .020
    diag.start_stages()
    clock.wall = .030; clock.cpu = .006
    diag.checkpoint('v2x_map')
    clock.wall = .045; clock.cpu = .010
    diag.finish()
    assert diag.samples['stage_v2x_map_wall_ms'][0] == pytest.approx(10.)
    assert diag.samples['stage_v2x_map_thread_cpu_ms'][0] == pytest.approx(6.)
    assert diag.samples['stage_tail_or_early_return_wall_ms'][0] == pytest.approx(15.)
    assert diag.samples['stage_tail_or_early_return_thread_cpu_ms'][0] == pytest.approx(4.)


def test_detail_wrapper_preserves_result_exception_and_resets(monkeypatch):
    clock = SimpleNamespace(wall=0., cpu=0.)
    monkeypatch.setattr(module, 'time', SimpleNamespace(
        perf_counter=lambda: clock.wall, thread_time=lambda: clock.cpu,
        process_time=lambda: clock.cpu))
    diag = module.RuntimeDiagnostics(.025, 1.)
    result = object()
    def operation(fail=False):
        clock.wall += .020
        clock.cpu += .005
        if fail:
            raise ValueError('unchanged')
        return result
    obj = SimpleNamespace(operation=operation)
    diag.instrument(obj, 'operation', 'operation')
    diag.instrument(obj, 'operation', 'operation')  # no double wrapping on restart
    assert obj.operation() is result
    assert not diag.details
    diag.begin(0.)
    assert obj.operation() is result
    with pytest.raises(ValueError, match='unchanged'):
        obj.operation(fail=True)
    row = diag.details['operation']
    assert row['calls'] == 2
    assert row['exceptions'] == 1
    assert row['wall_total_ms'] == pytest.approx(40.)
    assert row['cpu_total_ms'] == pytest.approx(10.)
    diag.finish()
    clock.wall = 2.
    report = json.loads(diag.report({}).split(' ', 1)[1])
    assert report['detail_functions']['operation']['calls'] == 2
    assert not diag.details


def test_detail_excludes_callback_threads_and_uses_dynamic_role():
    import threading
    diag = module.RuntimeDiagnostics(.025)
    obj = SimpleNamespace(run=lambda: 7)
    role = ['live']
    diag.instrument(obj, 'run', lambda: role[0])
    diag.begin(0.)
    thread = threading.Thread(target=obj.run)
    thread.start()
    thread.join()
    assert not diag.details
    assert obj.run() == 7
    role[0] = 'probe'
    assert obj.run() == 7
    assert diag.details['live']['calls'] == diag.details['probe']['calls'] == 1


def test_real_mpc_prepare_details_leave_qp_unchanged():
    import numpy as np
    from .probe_support import configured_mpc
    mpc = configured_mpc('center')
    wp = mpc.model.reference_path.get_waypoint(30)
    mpc.model.update_states(wp.x, wp.y, wp.psi)
    mpc._init_problem(mpc.N, 0.)
    before = {k: mpc.optimizer._derivative_cache[k].copy() for k in ('q', 'l', 'u')}
    diag = module.RuntimeDiagnostics(.025)
    mpc._runtime_diagnostics = diag
    mpc._runtime_role = lambda: 'live_mpc'
    diag.instrument(mpc, '_init_problem', lambda: mpc._runtime_role() + '._init_problem')
    diag.begin(0.)
    mpc._init_problem(mpc.N, 0.)
    for k, value in before.items():
        np.testing.assert_allclose(mpc.optimizer._derivative_cache[k], value)
    assert diag.details['live_mpc._init_problem']['calls'] == 1
    phases = [k for k in diag.details if k.startswith('live_mpc.prepare.')]
    assert len(phases) == 7
    assert all(diag.details[k]['calls'] == 1 for k in phases)
