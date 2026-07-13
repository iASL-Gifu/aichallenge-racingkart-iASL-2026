import json

log_file = "/home/takenoyama/.gemini/antigravity/brain/50a37863-ebf1-454b-9b56-d98f3bfbdb79/.system_generated/logs/overview.txt"
with open(log_file, "r") as f:
    for line in f:
        if '"step_index":447' in line:
            data = json.loads(line)
            replacement = data["tool_calls"][0]["args"]["ReplacementContent"]
            print(replacement)
