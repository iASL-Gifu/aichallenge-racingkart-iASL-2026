import re

log_path = "/home/haruki/aichallenge-racingkart/output/20260709-154804/d1/autoware.log"

wp_pattern = re.compile(r"Switching to lane|Switching back to free|wp_id|N=\d+.*wp_id=(\d+)")

# Let's search for the actual waypoints in the log lines
with open(log_path, "r") as f:
    lines = f.readlines()

print("--- Analysis of solver failures ---")
current_wp = None
for i, line in enumerate(lines):
    # Track the current waypoint if printed
    wp_match = re.search(r"wp_id=(\d+)|wp=(\d+)|wp:(\d+)", line)
    if wp_match:
        current_wp = next(item for item in wp_match.groups() if item is not None)
    
    if "EMERGENCY: Margin reduced to 0.0" in line:
        # Look around for waypoint info or print line number
        print(f"Line {i+1}: EMERGENCY (Reduced to 0.0) at wp={current_wp}")
    elif "primal infeasible" in line:
        print(f"Line {i+1}: PRIMAL INFEASIBLE at wp={current_wp}")
    elif "maximum iterations reached" in line:
        print(f"Line {i+1}: MAX ITERATIONS at wp={current_wp}")
    elif "Switching" in line:
        print(f"Line {i+1}: {line.strip()} at wp={current_wp}")
