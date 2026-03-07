# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 15:19:54 2026

@author: clher
"""

# Save as: scripts/run_tests.py

"""
Run tests from within Spyder.
Execute this script to run all tests.
"""


# Get project root
import subprocess
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent

# Run pytest
result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
    cwd=project_root,
    capture_output=False
)

sys.exit(result.returncode)
