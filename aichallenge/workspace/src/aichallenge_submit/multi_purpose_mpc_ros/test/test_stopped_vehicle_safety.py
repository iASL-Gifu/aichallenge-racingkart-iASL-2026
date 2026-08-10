"""Tests for stopped-lead overtake and reverse-wait decisions."""

import unittest

from multi_purpose_mpc_ros.v2x_vehicle_tracker import (
    absolute_heading_difference,
    build_closed_path_arc_lengths,
    classify_prepass_timeout_reasons,
    classify_lane_conflicts,
    circular_forward_progress,
    continuous_condition_confirmed,
    evaluate_stopped_lead_overtake,
    follow_stop_deadlock_conditions_met,
    is_follow_target_ahead,
    is_follow_retry_within_distance,
    is_prepass_fallback_lane_change,
    lane_conflicts_are_clear,
    lateral_vehicle_clearance,
    longitudinal_vehicle_clearance,
    is_parallel_vehicle,
    ordered_outer_lane_candidates,
    ordered_prepass_fallback_candidates,
    prepass_recovery_owns_lane_selection,
    prediction_clears_moving_vehicle,
    project_to_closed_path_arc,
    project_to_closed_path_frenet,
    relative_longitudinal_distance,
    reverse_path_has_vehicle_conflict,
    select_latched_overtake_lane,
    select_parallel_abort_lane,
    select_safe_outer_lane,
    should_release_latched_overtake_lane,
    should_reevaluate_follow_overtake,
    should_count_mpc_recovery_success,
    should_recover_from_mpc_stall,
    should_hold_follow_escape_exclusive,
    should_release_active_overtake_distance_gate,
    should_reset_overtake_latch_for_target_change,
    should_reset_motion_latch,
    should_suppress_overtake_before_grounded_snapshot,
    should_release_prepass_distance_gate,
    should_start_l1_recovery_from_safety,
    should_start_prepass_recovery_from_safety,
    signed_closed_path_arc_distance,
    should_exit_l1_probe_backoff,
    update_continuous_condition_since,
    update_fallback_commit_success_since,
    update_follow_escape_probe_success_cycles,
    update_motion_latch,
    startup_follow_restart_gap,
    startup_same_lane_lead_key,
)


class StoppedVehicleSafetyTest(unittest.TestCase):
    def test_reverse_corridor_ignores_adjacent_lane_vehicle(self):
        self.assertFalse(reverse_path_has_vehicle_conflict(
            ego_x=0.0, ego_y=0.0, ego_heading=0.0,
            reverse_distance=5.0, corridor_half_width=1.5,
            reverse_path_lanes={0}, vehicle_x=-2.0, vehicle_y=1.0,
            vehicle_lane=1,
        ))

    def test_reverse_corridor_blocks_same_lane_vehicle_on_swept_path(self):
        self.assertTrue(reverse_path_has_vehicle_conflict(
            ego_x=0.0, ego_y=0.0, ego_heading=0.0,
            reverse_distance=5.0, corridor_half_width=1.5,
            reverse_path_lanes={0}, vehicle_x=-2.0, vehicle_y=0.3,
            vehicle_lane=0,
        ))

    def test_reverse_corridor_checks_lateral_offset_for_unknown_lane(self):
        self.assertFalse(reverse_path_has_vehicle_conflict(
            ego_x=0.0, ego_y=0.0, ego_heading=0.0,
            reverse_distance=5.0, corridor_half_width=1.5,
            reverse_path_lanes={0}, vehicle_x=-2.0, vehicle_y=2.0,
            vehicle_lane=None,
        ))

    def test_motion_latch_survives_an_ordinary_stop(self):
        self.assertTrue(update_motion_latch(True, 0.0))

    def test_motion_latch_sets_only_after_vehicle_has_moved(self):
        self.assertFalse(update_motion_latch(False, 1.0))
        self.assertTrue(update_motion_latch(False, 1.01))

    def test_motion_latch_reset_is_limited_to_pre_start_states(self):
        self.assertTrue(should_reset_motion_latch("Grounded"))
        self.assertTrue(should_reset_motion_latch("Ready"))
        self.assertFalse(should_reset_motion_latch("Start"))
        self.assertFalse(should_reset_motion_latch(None))

    def test_overtake_waits_for_grounded_snapshot(self):
        for state in (None, "Grounded", "Ready"):
            self.assertTrue(
                should_suppress_overtake_before_grounded_snapshot(
                    state, snapshot_completed=False))
        self.assertFalse(should_suppress_overtake_before_grounded_snapshot(
            "Grounded", snapshot_completed=True))

    def test_missed_snapshot_does_not_disable_entire_race(self):
        self.assertFalse(should_suppress_overtake_before_grounded_snapshot(
            "Start", snapshot_completed=False))

    def test_startup_restart_uses_separate_shorter_gap(self):
        self.assertEqual(startup_follow_restart_gap(
            startup_waiting=True,
            normal_min_gap=6.0,
            startup_min_gap=4.0,
        ), 4.0)
        self.assertEqual(startup_follow_restart_gap(
            startup_waiting=False,
            normal_min_gap=6.0,
            startup_min_gap=4.0,
        ), 6.0)

    def test_startup_target_prefers_same_lane_forward_vehicle(self):
        d2_key = startup_same_lane_lead_key(
            vehicle_id="d2", vehicle_lane_idx=0, ego_lane_idx=2,
            longitudinal=1.6, distance=3.7)
        d3_key = startup_same_lane_lead_key(
            vehicle_id="d3", vehicle_lane_idx=2, ego_lane_idx=2,
            longitudinal=4.45, distance=4.46)
        self.assertIsNone(d2_key)
        self.assertEqual(d3_key, (4.45, 4.46, "d3"))

    def test_startup_target_chooses_nearest_forward_same_lane_vehicle(self):
        near = startup_same_lane_lead_key(
            vehicle_id="near", vehicle_lane_idx=2, ego_lane_idx=2,
            longitudinal=4.0, distance=4.2)
        far = startup_same_lane_lead_key(
            vehicle_id="far", vehicle_lane_idx=2, ego_lane_idx=2,
            longitudinal=7.0, distance=7.1)
        behind = startup_same_lane_lead_key(
            vehicle_id="behind", vehicle_lane_idx=2, ego_lane_idx=2,
            longitudinal=-1.0, distance=1.0)
        self.assertLess(near, far)
        self.assertIsNone(behind)

    def evaluate(self, **overrides):
        values = {
            "tracked_stopped_lead": True,
            "distance": 5.0,
            "left_is_free": True,
            "right_is_free": False,
            "target_lane_idx": 2,
            "infeasibility_counter": 0,
            "reverse_distance": 3.0,
            "infeasible_cycles": 5,
        }
        values.update(overrides)
        return evaluate_stopped_lead_overtake(**values)

    def test_outer_lane_accepted_enables_forced_overtake(self):
        self.assertEqual(self.evaluate(), (True, False))

    def test_close_lead_without_passing_space_requests_reverse(self):
        self.assertEqual(self.evaluate(
            distance=2.5,
            left_is_free=False,
            target_lane_idx=1,
        ), (False, True))

    def test_close_lead_without_accepted_outer_lane_requests_reverse(self):
        self.assertEqual(self.evaluate(
            distance=2.5,
            target_lane_idx=1,
        ), (False, True))

    def test_repeated_infeasibility_requests_reverse(self):
        self.assertEqual(self.evaluate(
            distance=2.5,
            infeasibility_counter=5,
        ), (False, True))

    def test_moving_or_untracked_vehicle_gets_no_exemption(self):
        self.assertEqual(self.evaluate(
            tracked_stopped_lead=False,
            distance=2.5,
        ), (False, False))

    def test_follow_target_must_be_ahead(self):
        self.assertTrue(is_follow_target_ahead(0.01))
        self.assertFalse(is_follow_target_ahead(0.0))
        self.assertFalse(is_follow_target_ahead(-0.01))
        self.assertFalse(is_follow_target_ahead(None))

    def test_follow_deadlock_requires_every_stop_condition(self):
        values = {
            "follow_active": True,
            "ego_speed": 0.0,
            "lead_speed": 0.0,
            "gnss_moved_distance": 0.0,
            "forward_command": 0.0,
            "ego_speed_threshold": 0.15,
            "lead_speed_threshold": 0.3,
            "gnss_distance_threshold": 0.3,
            "forward_command_threshold": 0.3,
        }
        self.assertTrue(follow_stop_deadlock_conditions_met(**values))
        values["lead_speed"] = 0.3
        self.assertFalse(follow_stop_deadlock_conditions_met(**values))

    def test_forward_escape_probe_requires_three_consecutive_safe_cycles(self):
        cycles = 0
        for expected in (1, 2, 3):
            cycles = update_follow_escape_probe_success_cycles(
                cycles,
                lane_applied=True,
                feasible_solution=True,
                executable_forward_prediction=True,
                prediction_clear=True,
                emergency_brake_active=False,
            )
            self.assertEqual(cycles, expected)
        cycles = update_follow_escape_probe_success_cycles(
            cycles,
            lane_applied=True,
            feasible_solution=True,
            executable_forward_prediction=True,
            prediction_clear=False,
            emergency_brake_active=False,
        )
        self.assertEqual(cycles, 0)

    def test_new_stopped_target_discards_previous_overtake_latch(self):
        self.assertTrue(should_reset_overtake_latch_for_target_change(
            active_target_id="d3",
            candidate_target_id="d1",
            candidate_is_relevant=True,
        ))
        self.assertFalse(should_reset_overtake_latch_for_target_change(
            active_target_id="d1",
            candidate_target_id="d1",
            candidate_is_relevant=True,
        ))
        self.assertFalse(should_reset_overtake_latch_for_target_change(
            active_target_id="d3",
            candidate_target_id="d1",
            candidate_is_relevant=False,
        ))

    def test_deadlock_escape_holds_until_pass_or_safe_distance(self):
        self.assertTrue(should_hold_follow_escape_exclusive(
            target_longitudinal=3.0,
            target_distance=3.5,
            safe_distance=8.0,
        ))
        self.assertFalse(should_hold_follow_escape_exclusive(
            target_longitudinal=-0.1,
            target_distance=1.0,
            safe_distance=8.0,
        ))
        self.assertFalse(should_hold_follow_escape_exclusive(
            target_longitudinal=8.0,
            target_distance=8.0,
            safe_distance=8.0,
        ))


class PrepassSafetyRecoveryTest(unittest.TestCase):
    def test_recovery_success_is_not_counted_during_reverse(self):
        self.assertFalse(should_count_mpc_recovery_success(
            stuck_recovery_active=True,
            gear_is_drive=False,
            infeasibility_counter=0,
            has_current_prediction=True,
            used_prediction_fallback=False,
        ))

    def test_recovery_success_requires_confirmed_drive(self):
        self.assertFalse(should_count_mpc_recovery_success(
            stuck_recovery_active=False,
            gear_is_drive=False,
            infeasibility_counter=0,
            has_current_prediction=True,
            used_prediction_fallback=False,
        ))
        self.assertTrue(should_count_mpc_recovery_success(
            stuck_recovery_active=False,
            gear_is_drive=True,
            infeasibility_counter=0,
            has_current_prediction=True,
            used_prediction_fallback=False,
        ))

    def test_fresh_feasible_prediction_prevents_mpc_stall_reverse(self):
        self.assertFalse(should_recover_from_mpc_stall(
            safety_recovery_active=True,
            actual_speed=0.0,
            stall_speed_threshold=0.4,
            gnss_is_stuck=True,
            infeasibility_counter=0,
            has_fresh_valid_prediction=True,
        ))

    def test_missing_prediction_allows_mpc_stall_reverse(self):
        self.assertTrue(should_recover_from_mpc_stall(
            safety_recovery_active=True,
            actual_speed=0.0,
            stall_speed_threshold=0.4,
            gnss_is_stuck=True,
            infeasibility_counter=4,
            has_fresh_valid_prediction=False,
        ))

    def test_follow_overtake_is_reevaluated_every_two_seconds(self):
        self.assertFalse(should_reevaluate_follow_overtake(
            True, 11.9, 10.0, 2.0))
        self.assertTrue(should_reevaluate_follow_overtake(
            True, 12.0, 10.0, 2.0))
        self.assertFalse(should_reevaluate_follow_overtake(
            False, 20.0, 10.0, 2.0))

    def test_follow_retry_distance_gate_is_strictly_below_ten_metres(self):
        self.assertTrue(is_follow_retry_within_distance(9.999, 10.0))
        self.assertFalse(is_follow_retry_within_distance(10.0, 10.0))
        self.assertFalse(is_follow_retry_within_distance(10.001, 10.0))

    def test_active_outer_lane_releases_when_ahead_target_exits_gate(self):
        self.assertFalse(should_release_active_overtake_distance_gate(
            outer_lane_active=True,
            target_longitudinal=9.0,
            distance=9.999,
            max_distance=10.0,
        ))
        self.assertTrue(should_release_active_overtake_distance_gate(
            outer_lane_active=True,
            target_longitudinal=10.0,
            distance=10.0,
            max_distance=10.0,
        ))
        self.assertFalse(should_release_active_overtake_distance_gate(
            outer_lane_active=True,
            target_longitudinal=-0.1,
            distance=12.0,
            max_distance=10.0,
        ))
        self.assertFalse(should_release_active_overtake_distance_gate(
            outer_lane_active=False,
            target_longitudinal=12.0,
            distance=12.0,
            max_distance=10.0,
        ))

    def test_active_prepass_releases_immediately_at_distance_gate(self):
        self.assertFalse(should_release_prepass_distance_gate(
            recovery_active=True, distance=9.999, max_distance=10.0))
        self.assertTrue(should_release_prepass_distance_gate(
            recovery_active=True, distance=10.0, max_distance=10.0))
        self.assertTrue(should_release_prepass_distance_gate(
            recovery_active=True, distance=12.0, max_distance=10.0))

    def test_missing_distance_does_not_confirm_prepass_gate_exit(self):
        self.assertFalse(should_release_prepass_distance_gate(
            recovery_active=True, distance=None, max_distance=10.0))
        self.assertFalse(should_release_prepass_distance_gate(
            recovery_active=False, distance=12.0, max_distance=10.0))

    def test_prepass_behind_release_requires_continuous_confirmation(self):
        since = update_continuous_condition_since(
            None, now_sec=10.0, condition=True)
        self.assertFalse(continuous_condition_confirmed(
            since, now_sec=10.24, confirm_sec=0.25))
        self.assertTrue(continuous_condition_confirmed(
            since, now_sec=10.25, confirm_sec=0.25))
        self.assertIsNone(update_continuous_condition_since(
            since, now_sec=10.2, condition=False))

    def test_timeout_reasons_are_machine_readable(self):
        self.assertEqual(classify_prepass_timeout_reasons(
            mpc_stable=False,
            heading_stable=False,
            dynamics_stable=True,
            physical_passage_available=True,
            traffic_clear=False,
            distance_within_gate=True,
            target_behind=False,
        ), ("mpc_unstable", "heading_unstable", "traffic_blocked"))

    def test_timeout_reports_physical_block_separately_from_traffic(self):
        self.assertEqual(classify_prepass_timeout_reasons(
            mpc_stable=True,
            heading_stable=True,
            dynamics_stable=True,
            physical_passage_available=False,
            traffic_clear=False,
            distance_within_gate=True,
            target_behind=False,
        ), ("physical_passage_blocked",))

    def test_fallback_commit_tracks_continuous_success_time(self):
        success_since = update_fallback_commit_success_since(
            None, now_sec=10.0,
            lane_applied=True, feasible_solution=True)
        self.assertEqual(success_since, 10.0)
        self.assertEqual(update_fallback_commit_success_since(
            success_since, now_sec=10.3,
            lane_applied=True, feasible_solution=True), 10.0)
        self.assertIsNone(update_fallback_commit_success_since(
            success_since, now_sec=10.2,
            lane_applied=True, feasible_solution=False))
        self.assertIsNone(update_fallback_commit_success_since(
            success_since, now_sec=10.2,
            lane_applied=False, feasible_solution=True))

    def should_start(self, **overrides):
        values = {
            "recovery_requested": True,
            "fallback_enabled": True,
            "applied_lane_idx": 0,
            "latched_vehicle_id": "vehicle-a",
            "opponent_ahead": True,
            "fallback_recovery_active": False,
            "fallback_follow_active": False,
        }
        values.update(overrides)
        return should_start_prepass_recovery_from_safety(**values)

    def test_outer_lane_failure_starts_prepass_recovery(self):
        self.assertTrue(self.should_start())
        self.assertTrue(self.should_start(applied_lane_idx=2))

    def test_full_width_or_l1_failure_is_not_attributed_to_outer_lane(self):
        self.assertFalse(self.should_start(applied_lane_idx=None))
        self.assertFalse(self.should_start(applied_lane_idx=1))

    def test_missing_or_cleared_target_does_not_start_prepass_recovery(self):
        self.assertFalse(self.should_start(latched_vehicle_id=None))
        self.assertFalse(self.should_start(opponent_ahead=False))

    def test_existing_fallback_state_is_not_restarted(self):
        self.assertFalse(self.should_start(fallback_recovery_active=True))
        self.assertFalse(self.should_start(fallback_follow_active=True))

    def test_failed_lane_is_excluded_from_retry_candidates(self):
        self.assertEqual(ordered_outer_lane_candidates(0, 0), (2,))
        self.assertEqual(ordered_outer_lane_candidates(2, 2), (0,))

    def test_normal_retry_still_checks_opposite_lane_first(self):
        self.assertEqual(ordered_outer_lane_candidates(0), (2, 0))
        self.assertEqual(ordered_outer_lane_candidates(2), (0, 2))

    def test_fallback_reprobes_failed_lane_before_l1(self):
        self.assertEqual(
            ordered_prepass_fallback_candidates(0), (2, 0, 1))
        self.assertEqual(
            ordered_prepass_fallback_candidates(2), (0, 2, 1))

    def test_failed_fallback_attempts_advance_toward_l1(self):
        self.assertEqual(
            ordered_prepass_fallback_candidates(0, {2}), (0, 1))
        self.assertEqual(
            ordered_prepass_fallback_candidates(0, {0, 2}), (1,))


class L1SafetyRecoveryTest(unittest.TestCase):
    def test_circular_waypoint_progress_handles_lap_wrap(self):
        self.assertEqual(circular_forward_progress(310, 3, 313), 6)
        self.assertEqual(circular_forward_progress(20, 25, 313), 5)

    def test_l1_backoff_requires_all_release_conditions(self):
        ready = {
            "elapsed_sec": 2.0,
            "cooldown_sec": 2.0,
            "waypoint_progress": 5,
            "minimum_waypoint_progress": 5,
            "full_width_success_sec": 0.5,
            "required_full_width_success_sec": 0.5,
        }
        self.assertTrue(should_exit_l1_probe_backoff(**ready))
        for key, value in (
            ("elapsed_sec", 1.99),
            ("waypoint_progress", 4),
            ("full_width_success_sec", 0.49),
        ):
            blocked = dict(ready)
            blocked[key] = value
            self.assertFalse(should_exit_l1_probe_backoff(**blocked))

    def test_applied_l1_failure_starts_direct_recovery(self):
        self.assertTrue(should_start_l1_recovery_from_safety(
            recovery_requested=True,
            applied_lane_idx=1,
            l1_recovery_pending=False,
        ))

    def test_full_width_and_outer_lane_failures_are_not_l1_failures(self):
        for lane_idx in (None, 0, 2):
            self.assertFalse(should_start_l1_recovery_from_safety(
                recovery_requested=True,
                applied_lane_idx=lane_idx,
                l1_recovery_pending=False,
            ))

    def test_transition_request_without_applied_l1_is_not_attributed(self):
        self.assertFalse(should_start_l1_recovery_from_safety(
            recovery_requested=False,
            applied_lane_idx=1,
            l1_recovery_pending=False,
        ))

    def test_pending_l1_recovery_is_not_restarted(self):
        self.assertFalse(should_start_l1_recovery_from_safety(
            recovery_requested=True,
            applied_lane_idx=1,
            l1_recovery_pending=True,
        ))


class PrepassFallbackLaneChangeTest(unittest.TestCase):
    def test_recovery_exclusively_owns_lane_selection(self):
        self.assertTrue(prepass_recovery_owns_lane_selection(True))
        self.assertTrue(prepass_recovery_owns_lane_selection(
            False, fallback_commit_pending=True))
        self.assertFalse(prepass_recovery_owns_lane_selection(False))

    def test_selected_fallback_lane_bypasses_cooldown(self):
        for lane_idx in (0, 1, 2):
            self.assertTrue(is_prepass_fallback_lane_change(
                fallback_lane_idx=lane_idx,
                requested_lane_idx=lane_idx,
            ))

    def test_normal_or_different_lane_does_not_bypass_cooldown(self):
        self.assertFalse(is_prepass_fallback_lane_change(
            fallback_lane_idx=None,
            requested_lane_idx=0,
        ))
        self.assertFalse(is_prepass_fallback_lane_change(
            fallback_lane_idx=2,
            requested_lane_idx=0,
        ))


class OvertakeGeometryTest(unittest.TestCase):
    def parallel(self, **overrides):
        values = {
            "ego_lane_idx": 0,
            "other_lane_idx": 1,
            "lateral_clearance": 0.5,
            "longitudinal_clearance": -1.0,
            "longitudinal_distance": 1.0,
            "maximum_lateral_clearance": 1.3,
            "maximum_longitudinal_clearance": 0.5,
            "minimum_longitudinal_distance": -0.5,
            "maximum_longitudinal_distance": 4.5,
        }
        values.update(overrides)
        return is_parallel_vehicle(**values)

    def test_parallel_requires_different_lanes(self):
        self.assertFalse(self.parallel(other_lane_idx=0))

    def test_parallel_has_no_minimum_lateral_clearance(self):
        self.assertTrue(self.parallel(lateral_clearance=-0.2))
        self.assertTrue(self.parallel(lateral_clearance=0.0))
        self.assertTrue(self.parallel(lateral_clearance=1.3))
        self.assertFalse(self.parallel(lateral_clearance=1.301))

    def test_parallel_uses_vehicle_envelope_clearance(self):
        self.assertAlmostEqual(
            lateral_vehicle_clearance(1.525, 1.6, 0.725), 0.0)
        self.assertAlmostEqual(
            lateral_vehicle_clearance(1.4, 1.6, 0.725), -0.125)
        self.assertAlmostEqual(
            lateral_vehicle_clearance(2.0, 1.6, 0.725), 0.475)
        self.assertAlmostEqual(
            longitudinal_vehicle_clearance(2.0, 1.0, 1.0), 0.0)
        self.assertAlmostEqual(
            longitudinal_vehicle_clearance(4.35, 1.0, 1.0), 2.35)

    def test_parallel_rejects_longitudinally_separated_envelopes(self):
        self.assertTrue(self.parallel(longitudinal_clearance=0.5))
        self.assertFalse(self.parallel(longitudinal_clearance=0.501))
        # The false-positive observed in the D2 log: 2 m-long vehicles whose
        # centers are 4.35 m apart retain 2.35 m of longitudinal free space.
        self.assertFalse(self.parallel(
            longitudinal_distance=4.35,
            longitudinal_clearance=longitudinal_vehicle_clearance(
                4.35, 1.0, 1.0),
        ))

    def test_center_frenet_lateral_offsets_are_projected_independently(self):
        points, cumulative, total = build_closed_path_arc_lengths([
            (0.0, 0.0),
            (10.0, 0.0),
            (10.0, 10.0),
            (0.0, 10.0),
        ])
        ego = project_to_closed_path_frenet(
            5.0, 1.0, points, cumulative, total)
        other = project_to_closed_path_frenet(
            5.0, 2.5, points, cumulative, total)
        self.assertIsNotNone(ego)
        self.assertIsNotNone(other)
        self.assertAlmostEqual(ego[1], 1.0)
        self.assertAlmostEqual(other[1], 2.5)
        self.assertAlmostEqual(abs(other[1] - ego[1]), 1.5)

    def test_parallel_abort_keeps_outer_lane_when_opponent_is_l1(self):
        self.assertEqual(select_parallel_abort_lane(0, 1), 0)
        self.assertEqual(select_parallel_abort_lane(2, 1), 2)
        self.assertEqual(select_parallel_abort_lane(0, 2), 1)

    def test_parallel_requires_small_longitudinal_gap(self):
        self.assertFalse(self.parallel(longitudinal_distance=-0.51))
        self.assertTrue(self.parallel(longitudinal_distance=-0.5))
        self.assertTrue(self.parallel(longitudinal_distance=4.5))
        self.assertFalse(self.parallel(longitudinal_distance=4.51))

    def test_parallel_uses_signed_longitudinal_distance(self):
        self.assertFalse(self.parallel(longitudinal_distance=-4.0))
        self.assertTrue(self.parallel(longitudinal_distance=4.0))

    def test_parallel_rejects_unknown_lane(self):
        self.assertFalse(self.parallel(ego_lane_idx=None))

    def test_center_arc_filter_wraps_across_path_boundary(self):
        points, cumulative, total = build_closed_path_arc_lengths([
            (0.0, 0.0),
            (10.0, 0.0),
            (10.0, 10.0),
            (0.0, 10.0),
        ])
        ego_s = project_to_closed_path_arc(
            0.0, 0.2, points, cumulative, total)
        ahead_s = project_to_closed_path_arc(
            0.2, 0.0, points, cumulative, total)
        self.assertAlmostEqual(
            signed_closed_path_arc_distance(ego_s, ahead_s, total),
            0.4,
            places=6,
        )

    def test_center_arc_filter_preserves_signed_rear_distance(self):
        points, cumulative, total = build_closed_path_arc_lengths([
            (0.0, 0.0),
            (10.0, 0.0),
            (10.0, 10.0),
            (0.0, 10.0),
        ])
        ego_s = project_to_closed_path_arc(
            1.0, 0.0, points, cumulative, total)
        rear_s = project_to_closed_path_arc(
            0.4, 0.0, points, cumulative, total)
        self.assertAlmostEqual(
            signed_closed_path_arc_distance(ego_s, rear_s, total),
            -0.6,
            places=6,
        )

    def test_prediction_must_clear_moving_target_at_every_step(self):
        self.assertFalse(prediction_clears_moving_vehicle(
            [0.0, 2.0, 4.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 2.0],
            vehicle_x=2.0,
            vehicle_y=0.5,
            vehicle_vx=0.0,
            vehicle_vy=0.0,
            minimum_clearance=1.3,
        ))

    def test_prediction_can_bypass_brake_when_all_steps_are_clear(self):
        self.assertTrue(prediction_clears_moving_vehicle(
            [0.0, 2.0, 4.0],
            [2.0, 2.0, 2.0],
            [0.0, 1.0, 2.0],
            vehicle_x=2.0,
            vehicle_y=0.0,
            vehicle_vx=0.0,
            vehicle_vy=0.0,
            minimum_clearance=1.3,
        ))

    def test_empty_prediction_never_bypasses_brake(self):
        self.assertFalse(prediction_clears_moving_vehicle(
            [], [], [],
            vehicle_x=0.0,
            vehicle_y=0.0,
            vehicle_vx=0.0,
            vehicle_vy=0.0,
            minimum_clearance=1.3,
        ))

    def test_lane_conflicts_cover_front_side_and_rear(self):
        conflicts = classify_lane_conflicts(
            2,
            [("front", 2, 6.0), ("side", 2, 1.0), ("rear", 2, -4.0),
             ("other-lane", 0, 0.0)],
            front_distance=8.0,
            side_distance=2.0,
            rear_distance=5.0,
        )
        self.assertEqual(conflicts, {
            "front": ["front"], "side": ["side"], "rear": ["rear"]})
        self.assertFalse(lane_conflicts_are_clear(conflicts))

    def test_lane_is_clear_when_only_other_lanes_are_occupied(self):
        conflicts = classify_lane_conflicts(
            1,
            [("left", 2, 1.0), ("right", 0, -1.0)],
            front_distance=8.0,
            side_distance=2.0,
            rear_distance=5.0,
        )
        self.assertTrue(lane_conflicts_are_clear(conflicts))

    def test_safe_outer_lane_prefers_requested_clear_side(self):
        self.assertEqual(select_safe_outer_lane(
            0,
            {0: True, 2: True},
            {
                0: {"front": [], "side": [], "rear": []},
                2: {"front": [], "side": [], "rear": []},
            },
        ), 0)

    def test_safe_outer_lane_uses_other_side_when_preferred_is_blocked(self):
        self.assertEqual(select_safe_outer_lane(
            0,
            {0: True, 2: True},
            {
                0: {"front": ["d1"], "side": [], "rear": []},
                2: {"front": [], "side": [], "rear": []},
            },
        ), 2)

    def test_safe_outer_lane_rejects_width_or_traffic_blockage(self):
        self.assertIsNone(select_safe_outer_lane(
            2,
            {0: False, 2: True},
            {
                0: {"front": [], "side": [], "rear": []},
                2: {"front": [], "side": ["d3"], "rear": []},
            },
        ))

    def test_vehicle_ahead_has_positive_longitudinal_distance(self):
        self.assertGreater(relative_longitudinal_distance(5.0, 0.0, 0.0), 0.0)

    def test_vehicle_behind_has_negative_longitudinal_distance(self):
        self.assertLess(relative_longitudinal_distance(-5.0, 0.0, 0.0), 0.0)

    def test_longitudinal_projection_respects_ego_heading(self):
        self.assertLess(
            relative_longitudinal_distance(0.0, -5.0, 1.5707963267948966),
            0.0,
        )

    def test_heading_difference_wraps_at_pi(self):
        difference = absolute_heading_difference(
            3.12413936106985,
            -3.12413936106985,
        )
        self.assertAlmostEqual(difference, 0.034906585039886195)

    def test_latched_passing_side_ignores_opposite_candidate(self):
        decision = select_latched_overtake_lane(
            True,
            "vehicle-b",
            0,
            "vehicle-a",
            2,
        )
        self.assertEqual(decision, (2, "vehicle-a", 2, False))

    def test_new_passing_side_is_latched_with_vehicle_id(self):
        decision = select_latched_overtake_lane(
            True,
            "vehicle-a",
            0,
            None,
            None,
        )
        self.assertEqual(decision, (0, "vehicle-a", 0, True))

    def test_latch_is_released_when_overtake_finishes(self):
        decision = select_latched_overtake_lane(
            False,
            "vehicle-b",
            0,
            "vehicle-a",
            2,
        )
        self.assertEqual(decision, (None, None, None, False))

    def test_outer_lane_releases_after_target_is_behind_and_mpc_stalls(self):
        self.assertTrue(should_release_latched_overtake_lane(
            overtake_active=True,
            latched_vehicle_id="vehicle-a",
            latched_lane_idx=0,
            target_longitudinal_distance=-1.5,
            infeasibility_counter=8,
            behind_distance=1.0,
            infeasible_cycles=8,
        ))

    def test_outer_lane_does_not_release_while_target_is_ahead(self):
        self.assertFalse(should_release_latched_overtake_lane(
            overtake_active=True,
            latched_vehicle_id="vehicle-a",
            latched_lane_idx=2,
            target_longitudinal_distance=0.5,
            infeasibility_counter=20,
            behind_distance=1.0,
            infeasible_cycles=8,
        ))

    def test_outer_lane_does_not_release_for_transient_infeasibility(self):
        self.assertFalse(should_release_latched_overtake_lane(
            overtake_active=True,
            latched_vehicle_id="vehicle-a",
            latched_lane_idx=0,
            target_longitudinal_distance=-2.0,
            infeasibility_counter=7,
            behind_distance=1.0,
            infeasible_cycles=8,
        ))

    def test_center_lane_is_never_released_by_outer_lane_escape(self):
        self.assertFalse(should_release_latched_overtake_lane(
            overtake_active=True,
            latched_vehicle_id="vehicle-a",
            latched_lane_idx=1,
            target_longitudinal_distance=-2.0,
            infeasibility_counter=8,
            behind_distance=1.0,
            infeasible_cycles=8,
        ))


if __name__ == "__main__":
    unittest.main()
