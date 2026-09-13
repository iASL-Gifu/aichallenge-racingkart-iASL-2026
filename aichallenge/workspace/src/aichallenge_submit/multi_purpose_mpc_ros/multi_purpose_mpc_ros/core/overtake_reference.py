"""Shared spatial entry reference for shadow admission and live application."""
import numpy as np
from multi_purpose_mpc_ros.core.MPC import spatial_lane_transition_reference, blend_previous_lateral_prediction


def transition_length(c, lateral_distance, speed):
    speed = max(float(speed), 0.)
    length = (c._hybrid_overtake_base_length
              + c._hybrid_overtake_offset_gain * abs(lateral_distance)
              + c._hybrid_overtake_speed_gain * speed)
    low_speed = speed <= c._hybrid_overtake_low_speed_start_threshold
    if low_speed:
        length = min(length, c._hybrid_overtake_low_speed_max_length)
    return float(np.clip(length, c._hybrid_overtake_min_length,
                         c._hybrid_overtake_max_length)), low_speed


def apply_transition_reference(c, mpc, wp, lane, distances, start, length, previous=None):
    centers = np.asarray([mpc._compute_lane_center(wp+n, lane)
                          for n in range(mpc.N+1)], dtype=float)
    if lane == 2 and c._l2_inward_offset_zones:
        centers = np.asarray(c._l2_inward_targets(wp), dtype=float)
    targets, weights = spatial_lane_transition_reference(distances, start, centers, length)
    continuity = bool(previous is not None and len(previous) == len(targets)
                      and np.max(np.abs(previous-targets))
                      <= c._hybrid_overtake_continuity_max_deviation)
    targets = blend_previous_lateral_prediction(targets, previous,
        c._hybrid_overtake_continuity_weight, c._hybrid_overtake_continuity_max_deviation)
    mpc.set_soft_lateral_reference(lane_idx=lane, start_e_y=start,
                                   alpha=1., lateral_targets=targets)
    mpc.set_lane_transition_weights(weights[1:])
    return targets, weights, continuity, centers


def prepare_entry_reference(c, mpc, lane, speed):
    # Private model pose has already been updated. No live session/progress mutation.
    wp = int(mpc.model.wp_id)
    start = float(mpc.model.spatial_state.e_y)
    length, _ = transition_length(c, mpc._compute_lane_center(wp, lane)-start, speed)
    path = mpc.model.reference_path
    distances = np.concatenate(([0.], np.cumsum([
        path.segment_lengths[(wp+n) % path.n_waypoints] for n in range(mpc.N)])))
    return apply_transition_reference(c, mpc, wp, lane, distances, start, length)
