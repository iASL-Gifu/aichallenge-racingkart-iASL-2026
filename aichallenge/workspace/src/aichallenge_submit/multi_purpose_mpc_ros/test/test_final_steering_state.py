"""Final recovery commands own the next live MPC steering-rate origin."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock
import numpy as np
import pytest
from .test_four_vehicle_deadlock import method
from .test_hybrid_integration import control_tree


def finalize(c, u):
    body = control_tree().body
    start = next(i for i,n in enumerate(body) if isinstance(n,ast.Assign)
                 and ast.unparse(n.targets[0]) == 'self._last_acc')
    end = next(i for i,n in enumerate(body) if isinstance(n,ast.Expr)
               and isinstance(n.value,ast.Call)
               and ast.unparse(n.value.func) == 'self._publish_control_command')
    exec(compile(ast.Module(body=body[start:end+1],type_ignores=[]),'<final>','exec'),
         dict(self=c,u=u,acc=0.,v=0.,now=None,bug_acc_enabled=False))


def next_steering(mpc):
    source = Path(__file__).parents[1] / 'multi_purpose_mpc_ros/core/MPC.py'
    fn = next(n for n in ast.walk(ast.parse(source.read_text()))
              if isinstance(n,ast.FunctionDef) and n.name == '_get_control_impl')
    block = next(n for n in ast.walk(fn) if isinstance(n,ast.Try))
    start = next(i for i,n in enumerate(block.body) if isinstance(n,ast.Assign)
                 and ast.unparse(n.targets[0]) == 'max_delta_change')
    scope = dict(self=mpc,delta=.3,np=np)
    exec(compile(ast.Module(body=block.body[start:start+3],type_ignores=[]),'<rate>','exec'),scope)
    return scope['delta']


@pytest.mark.parametrize('mode',['reverse','straight','curved_forward','smoothed','normal'])
def test_final_command_limits_next_live_solve(mode):
    mpc = NS(previous_steering=.2,max_steering_rate=2.,model=NS(Ts=.025))
    inactive = NS(previous_steering=-.1)
    c = NS(_mpc=mpc,_mpcN_race=inactive,_last_u=np.zeros(2),
           _car=NS(drive=Mock()),_runtime_checkpoint=Mock(),_publish_control_command=Mock(),
           _stuck_reverse_command_mode='negative_speed',_stuck_reverse_speed=.3,
           _stuck_reverse_steering_scale=0.)
    u = np.array([2.,.2])
    if mode == 'reverse':
        method('_apply_stuck_reverse_command')(c,u)
    elif mode in ('straight', 'curved_forward'):
        source = Path(__file__).parents[1] / 'multi_purpose_mpc_ros/mpc_controller.py'
        node = next(n for n in ast.walk(ast.parse(source.read_text()))
                    if isinstance(n,ast.FunctionDef) and n.name == '_apply_straight_reentry')
        override = next(n for n in node.body if isinstance(n,ast.Assign)
                        and ast.unparse(n.targets[0]) == 'u[1]')
        exec(compile(ast.Module(body=[override],type_ignores=[]),'<straight>','exec'),dict(u=u, motion=NS(direction=1, steering=.3 if mode == 'curved_forward' else 0.)))
    elif mode == 'smoothed':
        u[1] = .06
    expected = float(u[1])
    finalize(c,u)
    assert mpc.previous_steering == expected == c._last_u[1]
    assert c._publish_control_command.call_args.args[1][1] == expected
    assert inactive.previous_steering == -.1
    assert abs(next_steering(mpc)-expected) <= .05+1e-12
