import json
import subprocess
import sys

if len(sys.argv) != 2:
    print("Usage: python run_scenario.py <scenario.json>")
    sys.exit(1)

scenario_path = sys.argv[1]

with open(scenario_path, "r") as f:
    cfg = json.load(f)

subprocess.run(["python", "run_system.py", json.dumps(cfg)])
