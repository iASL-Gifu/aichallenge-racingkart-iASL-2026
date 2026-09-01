#!/usr/bin/env python3
"""Summarize AI Challenge AWSIM/controller logs without ROS dependencies."""

from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Optional


ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")
ROS_TIME_RE = re.compile(r"\[(\d{9,}(?:\.\d+)?)\]")
LAP_RE = re.compile(
    r"Lap\s+(\d+)\s+completed!\s+Lap time:\s*([0-9]+(?:\.[0-9]+)?)\s*s"
)
VEHICLE_ID_RE = re.compile(
    r"(?:vehicle_id|target_id|target)=([A-Za-z0-9_.:-]+)"
)

ATTEMPT_EVENT = "[OvertakeLatch] fixed passing side"
SUCCESS_EVENTS = (
    "[OvertakeTargetPhysicalComplete]",
    "forced overtake completed",
)
COMMIT_ABORT_EVENTS = (
    "[OvertakeLatchedLaneUnsafe]",
    "[PrepassShortCommitFailure]",
)
FAILURE_EVENTS = (
    "[OvertakeTargetChange]",
    "[PrepassDistanceGateRelease]",
)
OLD_VEHICLE_ID_RE = re.compile(r"old_vehicle_id=([A-Za-z0-9_.:-]+)")
# wall_at_step is commonly throttled near 1 Hz. Adjacent reports no more than
# one second apart describe one continuous unsafe interval, not new contacts.
WALL_UNSAFE_EPISODE_GAP_SEC = 1.0


def _timestamp(line: str) -> Optional[float]:
    match = ROS_TIME_RE.search(line)
    return float(match.group(1)) if match else None


def _vehicle_id(line: str) -> Optional[str]:
    match = VEHICLE_ID_RE.search(line)
    return match.group(1) if match else None


def _new_run(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "lap_times": [],
        "attempts": [],
        "hard_commits": 0,
        "lane_unsafe_events": 0,
        "short_commit_failure_events": 0,
        "recovery_durations": [],
        "recovery_entries": 0,
        "incomplete_recoveries": 0,
        "reverse_durations": [],
        "reverse_episodes": 0,
        "reverse_command_samples": 0,
        "incomplete_reverse": 0,
        "fallback_entries": 0,
        "osqp_primal_infeasible": 0,
        "safety": {
            "wall_unsafe_events": 0,
            "wall_unsafe_episodes": 0,
            "emergency_brake_events": 0,
            "lane_unsafe_events": 0,
            "outer_commit_corridor_blocked_events": 0,
            "prepass_soft_neutral_hold_entries": 0,
            "straight_reentry_starts": 0,
        },
    }


def analyze_file(path: Path) -> dict[str, Any]:
    """Parse one log. Episode state is intentionally not shared across runs."""
    result = _new_run(path)
    open_attempts: dict[str, dict[str, Any]] = {}
    recovery_start: Optional[float] = None
    reverse_start: Optional[float] = None
    last_wall_unsafe_time: Optional[float] = None

    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for raw_line in stream:
            line = ANSI_ESCAPE_RE.sub("", raw_line)
            now = _timestamp(line)

            lap_match = LAP_RE.search(line)
            if lap_match:
                result["lap_times"].append(float(lap_match.group(2)))

            if ATTEMPT_EVENT in line:
                result["hard_commits"] += 1
                vehicle_id = _vehicle_id(line)
                # Controller commit logs always carry vehicle_id. Retain a
                # stable fallback key for malformed/legacy log lines.
                episode_key = vehicle_id or "<unknown>"
                if episode_key not in open_attempts:
                    attempt = {
                        "vehicle_id": vehicle_id,
                        "start_time": now,
                        "outcome": "incomplete",
                        "outcome_event": None,
                    }
                    open_attempts[episode_key] = attempt
                    result["attempts"].append(attempt)

            success_event = next(
                (event for event in SUCCESS_EVENTS if event in line), None
            )
            if success_event:
                event_vehicle = _vehicle_id(line)
                episode_key = event_vehicle or "<unknown>"
                attempt = open_attempts.pop(episode_key, None)
                if attempt is not None:
                    attempt["outcome"] = "success"
                    attempt["outcome_event"] = success_event
                    attempt["end_time"] = now

            if "[OvertakeLatchedLaneUnsafe]" in line:
                result["lane_unsafe_events"] += 1
            if "[PrepassShortCommitFailure]" in line:
                result["short_commit_failure_events"] += 1

            failure_event = next(
                (event for event in FAILURE_EVENTS if event in line), None
            )
            if failure_event:
                if failure_event == "[OvertakeTargetChange]":
                    old_match = OLD_VEHICLE_ID_RE.search(line)
                    event_vehicle = old_match.group(1) if old_match else None
                else:
                    event_vehicle = _vehicle_id(line)
                episode_key = event_vehicle or "<unknown>"
                attempt = open_attempts.pop(episode_key, None)
                if attempt is not None:
                    attempt["outcome"] = "failure"
                    attempt["outcome_event"] = failure_event
                    attempt["end_time"] = now

            if "[MPCSafetyRecovery] discarding stale prediction" in line:
                if recovery_start is None:
                    recovery_start = now
                    result["recovery_entries"] += 1
            elif "[MPCSafetyRecovery] full-width MPC recovered" in line:
                if recovery_start is not None:
                    if recovery_start is not None and now is not None:
                        result["recovery_durations"].append(
                            max(0.0, now - recovery_start)
                        )
                    recovery_start = None

            if "[StraightReentryStart]" in line:
                result["safety"]["straight_reentry_starts"] += 1
                if "direction=REVERSE" in line and reverse_start is None:
                    reverse_start = now
                    result["reverse_episodes"] += 1
            if "[StuckRecovery] reverse cmd" in line:
                # This is a periodic command sample, not a new episode.
                result["reverse_command_samples"] += 1
            if (
                "[StraightReentryStop]" in line
                or "[StraightReentryComplete]" in line
            ):
                if reverse_start is not None:
                    if now is not None:
                        result["reverse_durations"].append(
                            max(0.0, now - reverse_start)
                        )
                    reverse_start = None

            if "[ActivePathSteeringFallbackEnter]" in line:
                result["fallback_entries"] += 1
            if "primal infeasible" in line.lower():
                result["osqp_primal_infeasible"] += 1
            if "wall_at_step" in line:
                result["safety"]["wall_unsafe_events"] += 1
                if (
                    now is None
                    or last_wall_unsafe_time is None
                    or now - last_wall_unsafe_time
                        > WALL_UNSAFE_EPISODE_GAP_SEC
                ):
                    result["safety"]["wall_unsafe_episodes"] += 1
                last_wall_unsafe_time = now
            if "[EmergencyBrake]" in line:
                result["safety"]["emergency_brake_events"] += 1
            if "[OvertakeLatchedLaneUnsafe]" in line:
                result["safety"]["lane_unsafe_events"] += 1
            if "[OuterCommitCorridorBlocked]" in line:
                result["safety"]["outer_commit_corridor_blocked_events"] += 1
            if "[PrepassSoftNeutralHoldEnter]" in line:
                result["safety"]["prepass_soft_neutral_hold_entries"] += 1

    if recovery_start is not None:
        result["incomplete_recoveries"] = 1
    if reverse_start is not None:
        result["incomplete_reverse"] = 1
    return result


def aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    lap_times = [value for run in runs for value in run["lap_times"]]
    attempts = [attempt for run in runs for attempt in run["attempts"]]
    recovery_durations = [
        value for run in runs for value in run["recovery_durations"]
    ]
    reverse_durations = [
        value for run in runs for value in run["reverse_durations"]
    ]
    successes = sum(attempt["outcome"] == "success" for attempt in attempts)
    failures = sum(attempt["outcome"] == "failure" for attempt in attempts)
    completed_attempts = successes + failures
    lane_unsafe_events = sum(run["lane_unsafe_events"] for run in runs)
    short_commit_failure_events = sum(
        run["short_commit_failure_events"] for run in runs
    )
    safety_keys = runs[0]["safety"].keys() if runs else ()
    safety = {
        key: sum(run["safety"][key] for run in runs) for key in safety_keys
    }
    safety["wall_unsafe_episode_gap_sec"] = WALL_UNSAFE_EPISODE_GAP_SEC

    return {
        "runs": len(runs),
        "files": [run["path"] for run in runs],
        "lap": {
            "lap_count": len(lap_times),
            "lap_times": lap_times,
            "best_lap": min(lap_times) if lap_times else None,
            "median_lap": statistics.median(lap_times) if lap_times else None,
            "worst_lap": max(lap_times) if lap_times else None,
        },
        "overtake": {
            "overtake_attempts": len(attempts),
            "hard_commits": sum(run["hard_commits"] for run in runs),
            "overtake_successes": successes,
            "overtake_failures": failures,
            "incomplete_overtake_attempts": sum(
                attempt["outcome"] == "incomplete" for attempt in attempts
            ),
            "commit_abort_events": (
                lane_unsafe_events + short_commit_failure_events
            ),
            "lane_unsafe_events": lane_unsafe_events,
            "short_commit_failure_events": short_commit_failure_events,
            "completed_success_rate": (
                100.0 * successes / completed_attempts
                if completed_attempts else None
            ),
            "attempt_success_rate": (
                100.0 * successes / len(attempts) if attempts else 0.0
            ),
            "attempt_event": ATTEMPT_EVENT,
            "success_events": list(SUCCESS_EVENTS),
            "commit_abort_events_used": list(COMMIT_ABORT_EVENTS),
            "failure_events": list(FAILURE_EVENTS),
        },
        "recovery": {
            "mpc_safety_recovery_entries": sum(
                run["recovery_entries"] for run in runs
            ),
            "recovery_episode_count": sum(
                run["recovery_entries"] for run in runs
            ),
            "recovery_durations": recovery_durations,
            "median_recovery_duration": (
                statistics.median(recovery_durations)
                if recovery_durations else None
            ),
            "max_recovery_duration": (
                max(recovery_durations) if recovery_durations else None
            ),
            "incomplete_recovery_count": sum(
                run["incomplete_recoveries"] for run in runs
            ),
            "active_path_steering_fallback_entries": sum(
                run["fallback_entries"] for run in runs
            ),
            "osqp_primal_infeasible_events": sum(
                run["osqp_primal_infeasible"] for run in runs
            ),
        },
        "reverse": {
            "reverse_episode_count": sum(
                run["reverse_episodes"] for run in runs
            ),
            "reverse_command_samples": sum(
                run["reverse_command_samples"] for run in runs
            ),
            "reverse_durations": reverse_durations,
            "total_reverse_duration": sum(reverse_durations),
            "max_reverse_duration": (
                max(reverse_durations) if reverse_durations else None
            ),
            "incomplete_reverse_count": sum(
                run["incomplete_reverse"] for run in runs
            ),
        },
        "safety": safety,
    }


def _duration(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.2f} s"


def human_summary(result: dict[str, Any]) -> str:
    lap = result["lap"]
    overtake = result["overtake"]
    recovery = result["recovery"]
    reverse = result["reverse"]
    safety = result["safety"]
    return "\n".join(
        [
            "# Race Evaluation",
            "",
            f"Runs : {result['runs']}",
            f"Completed laps : {lap['lap_count']}",
            "",
            "## Lap",
            "",
            f"Best : {_duration(lap['best_lap'])}",
            f"Median : {_duration(lap['median_lap'])}",
            f"Worst : {_duration(lap['worst_lap'])}",
            "",
            "## Overtake",
            "",
            f"Target attempts : {overtake['overtake_attempts']}",
            f"Hard commits : {overtake['hard_commits']}",
            f"Successes : {overtake['overtake_successes']}",
            f"Failures : {overtake['overtake_failures']}",
            f"Incomplete : {overtake['incomplete_overtake_attempts']}",
            f"Commit abort events : {overtake['commit_abort_events']}",
            f"Lane unsafe : {overtake['lane_unsafe_events']}",
            "Short commit failure : "
            f"{overtake['short_commit_failure_events']}",
            "Completed success rate : "
            + (
                f"{overtake['completed_success_rate']:.1f} %"
                if overtake["completed_success_rate"] is not None
                else "n/a"
            ),
            f"Attempt success rate : {overtake['attempt_success_rate']:.1f} %",
            "",
            "## Recovery",
            "",
            "MPC Safety Recovery : "
            f"{recovery['recovery_episode_count']}",
            "Median recovery : "
            f"{_duration(recovery['median_recovery_duration'])}",
            "Max recovery : "
            f"{_duration(recovery['max_recovery_duration'])}",
            "Incomplete recovery : "
            f"{recovery['incomplete_recovery_count']}",
            "Fallback entries : "
            f"{recovery['active_path_steering_fallback_entries']}",
            "OSQP primal infeasible : "
            f"{recovery['osqp_primal_infeasible_events']}",
            f"Reverse episodes : {reverse['reverse_episode_count']}",
            f"Reverse command samples : {reverse['reverse_command_samples']}",
            f"Total reverse : {_duration(reverse['total_reverse_duration'])}",
            f"Max reverse : {_duration(reverse['max_reverse_duration'])}",
            f"Incomplete reverse : {reverse['incomplete_reverse_count']}",
            "",
            "## Safety",
            "",
            f"wall unsafe events : {safety['wall_unsafe_events']}",
            f"wall unsafe episodes : {safety['wall_unsafe_episodes']}",
            f"EmergencyBrake : {safety['emergency_brake_events']}",
            f"LaneUnsafe : {safety['lane_unsafe_events']}",
            "OuterCommitCorridorBlocked : "
            f"{safety['outer_commit_corridor_blocked_events']}",
            "PrepassSoftNeutralHoldEnter : "
            f"{safety['prepass_soft_neutral_hold_entries']}",
            f"StraightReentryStart : {safety['straight_reentry_starts']}",
        ]
    )


def _expand_paths(arguments: list[str]) -> list[Path]:
    expanded: list[Path] = []
    for argument in arguments:
        matches = glob.glob(argument)
        expanded.extend(Path(match) for match in (matches or [argument]))
    return expanded


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize one or more AI Challenge AWSIM/controller logs."
    )
    parser.add_argument("logs", nargs="+", help="log paths or glob patterns")
    parser.add_argument(
        "--json", action="store_true", help="print machine-readable JSON"
    )
    args = parser.parse_args(argv)

    paths = _expand_paths(args.logs)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        parser.error("log file not found: " + ", ".join(missing))

    result = aggregate([analyze_file(path) for path in paths])
    if args.json:
        json.dump(result, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        print(human_summary(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
