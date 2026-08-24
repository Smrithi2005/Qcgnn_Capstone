import sys
from pathlib import Path

# Write to log file
log_file = Path("test_output.log")

with open(log_file, 'w') as f:
    f.write("TEST 1\n")
    f.write("TEST 2\n")
    f.write("TEST 3\n")
    f.write("If you see this, Python is working!\n")
    f.write(f"Python version: {sys.version}\n")
    f.write(f"Current directory: {Path.cwd()}\n")

print("Log file created")