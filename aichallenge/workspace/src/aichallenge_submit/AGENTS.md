# AGENTS.md

## Scope

These instructions apply to all work under this aichallenge_submit directory.

Read this file before investigating or modifying the controller.

## Project

This repository contains an autonomous racing kart controller based on
Autoware / ROS 2 Humble / MPC.

Main controller:

`multi_purpose_mpc_ros/multi_purpose_mpc_ros/mpc_controller.py`

The main goal is:

- complete all laps safely
- avoid vehicle-to-vehicle and wall collisions
- maintain competitive lap times
- perform overtaking using L0 / L1 / L2 lanes

## Development policy

Always follow this order:

1. Investigate logs and code
2. Identify one root cause
3. Propose one minimal change
4. Apply only that change
5. Run the same scenario again
6. Compare before / after

Do not stack multiple unrelated parameter or logic changes in one experiment.

## Investigation before modification

Before changing code:

- identify the exact failure interval
- reconstruct the state transition
- determine which controller owned steering and speed
- distinguish cause from downstream safety reactions

Do not assume that the component producing the final STOP or EBrake is the root cause.

For example:

unsafe lane transition
→ MPC instability
→ PP fallback
→ safety stop

means the PP safety stop may be correct and should not automatically be weakened.

## Visual observations

User visual observations are hypotheses, not confirmed facts.

Examples:

- "it looked like the cars collided"
- "it seemed to follow instead of overtake"
- "the opposite lane looked free"

Use these observations to locate relevant log intervals.

Always verify them against logs.

Clearly distinguish:

- confirmed by log
- highly likely
- possible
- visual observation only

## Collision judgement

Do not classify a vehicle collision from negative lateral clearance alone.

Vehicle collision evidence should consider both:

- longitudinal envelope clearance
- lateral envelope clearance

Prefer explicit simulator collision events if available.

Wall contact should be evaluated using physical boundary / vehicle-center allowable bounds.

## Lane terminology

L0 / L1 / L2 are the three usable lane/corridor regions.

Do not assume a fixed preferred lane.

When both L0 and L2 are available, investigate:

- physical passage
- front conflict
- side conflict
- rear conflict
- current clearance
- predicted clearance
- opponent lateral movement
- wall margin

The objective is not merely to find a safe lane, but to determine whether a safer
and more effective passing lane exists.
When multiple safe passing lanes exist, do not assume that any safe lane is equally good.
Investigate which lane provides the largest usable clearance and best predicted separation
from moving opponents while preserving wall safety.

## Safety philosophy

Do not weaken safety thresholds merely because lap time became slower.

First determine whether:

A. the safety rule is unnecessarily conservative

or

B. an upstream planning / MPC failure created a dangerous state where
the safety rule correctly had to activate.

Prefer fixing B before weakening A.

## Pure Pursuit fallback

Pure Pursuit fallback must not be treated as vehicle-safe solely because
the wall rollout is safe.

Nonzero PP motion requires:

- wall-safe rollout
- V2X-safe moving prediction

Existing PP V2X validation should not be removed without clear evidence.

## WallV2X

Keep the roles separate:

WallV2X:

- PP wall rollout is unsafe
- nearby V2X exists
- controlled pass is only allowed when explicitly proven safe

PP V2X validation:

- PP wall rollout is safe
- but moving vehicle prediction may still be unsafe

## MotionHold

Do not simply relax:

- heading threshold
- lateral speed threshold
- yaw rate threshold
- prediction freshness requirement

Investigate prediction generation / ownership / synchronization first.

## Current MotionHold validated prediction behavior

A MotionHold-triggered full-width solve may produce a validated prediction
for the same candidate on the immediately following control loop.

The validated result must not be reused for:

- another vehicle
- another lane
- stale loops
- lane-constrained prediction
- fallback prediction
- SafetyRecovery
- reverse
- another ownership mode

## Performance investigation

When lap time becomes worse, investigate where time is lost.

Possible categories:

- follow
- stop
- EmergencyBrake
- ParallelSafety
- SafetyRecovery
- MPC infeasible
- Pure Pursuit fallback
- reverse
- MotionHold
- failed overtake attempt

Do not infer the cause from event counts alone.

## Overtake investigation

For each overtake attempt, reconstruct:

- candidate
- commit
- lane transition
- physical pass
- release

For failures, identify the first decisive cause:

- no safe physical passage
- candidate conflict
- MotionHold
- lane unsafe
- MPC infeasible
- short commit failure
- follow fallback
- target switch
- distance gate
- other

## Change policy

Unless explicitly requested, do not modify multiple systems together.

Do not change these parameters casually:

- EmergencyBrake thresholds
- ParallelSafety thresholds
- MotionHold thresholds
- MPC horizon N
- global speed
- reverse behavior
- PP V2X safety thresholds

Prefer logic/root-cause fixes before threshold tuning.

## After each implementation

Always verify:

1. syntax / import correctness
2. expected new log event occurs
3. original failure path changes as intended
4. unrelated safety logic still works
5. safe cases retain previous performance
6. no new deadlock / follow / reverse loop appears

## Reporting format

When reporting investigation results, use:

### Conclusion

One sentence stating the main cause.

### Evidence

Relevant timestamps, waypoints and log events.

### Cause chain

Example:

candidate safe
→ commit blocked
→ opponent approaches
→ MPC becomes infeasible
→ fallback
→ safety stop

### Safety assessment

Explain whether each safety action was:

- necessary
- reasonable
- possibly conservative

### Next action

Recommend exactly one next investigation or one minimal change.