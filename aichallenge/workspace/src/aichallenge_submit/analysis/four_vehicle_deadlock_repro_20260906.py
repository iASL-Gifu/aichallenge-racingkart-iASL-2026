"""Entry point for the four repaired deadlock regression scenarios."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'multi_purpose_mpc_ros'))
from test.test_four_vehicle_deadlock import (
    test_free_l2_is_selected_by_follow_escape,
    test_abort_without_forward_group_member_releases_without_motion_permission,
    test_distant_fourth_car_leaves_safe_prefix_but_gets_no_creep_permission,
    test_rear_vehicle_shortens_reverse_with_body_and_braking_margin,
)
