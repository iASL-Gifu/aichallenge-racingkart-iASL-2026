import unittest

import numpy as np

from multi_purpose_mpc_ros.core.MPC import (
    build_arc_length_steering_reservation,
)


class SteeringReservationTest(unittest.TestCase):
    def test_ramp_starts_before_corner(self):
        distances = np.arange(0.0, 13.0, 1.0)
        references = np.zeros_like(distances)
        references[8:] = 0.30
        speeds = np.full_like(distances, 10.0)
        reserved = build_arc_length_steering_reservation(
            distances, references, speeds, 0.15, 0.60, 0.0, 0.025)
        self.assertGreater(reserved[3], 0.0)
        self.assertLessEqual(np.max(np.abs(np.diff(reserved))), 0.061)

    def test_first_command_obeys_physical_rate(self):
        distances = np.arange(0.0, 6.0, 1.0)
        references = np.full_like(distances, 0.30)
        speeds = np.full_like(distances, 10.0)
        reserved = build_arc_length_steering_reservation(
            distances, references, speeds, 0.15, 0.60, 0.0, 0.025)
        self.assertLessEqual(abs(reserved[0]), 0.60 * 0.025 + 1e-12)

    def test_delay_advances_future_demand(self):
        distances = np.arange(0.0, 11.0, 1.0)
        references = np.zeros_like(distances)
        references[6:] = 0.30
        speeds = np.full_like(distances, 10.0)
        without_delay = build_arc_length_steering_reservation(
            distances, references, speeds, 0.0, 0.60, 0.0, 0.025)
        with_delay = build_arc_length_steering_reservation(
            distances, references, speeds, 0.15, 0.60, 0.0, 0.025)
        self.assertGreaterEqual(with_delay[3], without_delay[3])


if __name__ == '__main__':
    unittest.main()


def test_reservation_changes_cost_not_vehicle_dynamics():
    from .probe_support import configured_mpc
    m = configured_mpc('race')
    p = m.model.reference_path
    wp = p.get_waypoint(242)
    m.model.update_states(wp.x, wp.y, wp.psi)
    m.model.spatial_state = m.model.t2s(wp, m.model.temporal_state)
    m.current_control = np.tile([35 / 3.6, 0.0], m.N)
    snapshots = []
    for enabled in (False, True):
        m.steering_reservation_enabled = enabled
        m.osqp_initialized = False
        m._init_problem(m.N, 0.)
        cache = m.optimizer._derivative_cache
        snapshots.append((m.A0.copy(), cache['l'].copy(), cache['u'].copy(), cache['q'].copy()))
    a, b = snapshots
    np.testing.assert_allclose(a[0].toarray(), b[0].toarray())
    np.testing.assert_allclose(a[1], b[1])
    np.testing.assert_allclose(a[2], b[2])
    assert not np.allclose(a[3], b[3])


def test_spatial_rate_bounds_guarantee_physical_angle_rate():
    from .probe_support import configured_mpc
    m = configured_mpc('center')
    p = m.model.reference_path
    wp = p.get_waypoint(248)
    m.model.update_states(wp.x, wp.y, wp.psi)
    m.model.spatial_state = m.model.t2s(wp, m.model.temporal_state)
    m.current_control = np.tile([35 / 3.6, 0.0], m.N)
    m._init_problem(m.N, 0.)
    start = 2 * m.nx_N + m.nu_N
    limits = m.optimizer._derivative_cache['u'][start:start + m.n_rate_constraints]
    kappa = np.linspace(-0.3, 0.3, 100)
    for i, limit in enumerate(limits):
        dt = float(p.get_waypoint(249+i) - p.get_waypoint(248+i)) / (35 / 3.6)
        angles = np.arctan(m.model.length * (kappa + limit)) - np.arctan(m.model.length * kappa)
        assert np.max(np.abs(angles)) / dt <= m.max_steering_rate + 1e-9
    # Spatial intervals here exceed the 25 ms controller period.
    assert np.all(limits > m.max_steering_rate * m.model.Ts)


def test_full_width_corner_converges_without_reducing_speed_setting():
    from .probe_support import configured_mpc
    for kind, index in [('race', 242), ('center', 249)]:
        m = configured_mpc(kind)
        wp = m.model.reference_path.get_waypoint(index)
        m.model.update_states(wp.x, wp.y, wp.psi)
        steer = np.arctan(m.model.length * wp.kappa * (1 + m.understeer_coeff * (35/3.6)**2))
        m.previous_steering = steer
        m.current_control = np.tile([35/3.6, steer], m.N)
        m.get_control()
        assert m.last_solution_accurate, m.failure_reason
        assert m.input_constraints['umax'][0] == 35/3.6
