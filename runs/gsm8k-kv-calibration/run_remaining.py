"""Continue the authorized serial experiment after calibration exits."""
import os
from pathlib import Path
import subprocess
import sys
import time

OUT=Path(__file__).resolve().parent
REPO=OUT.parents[1]
parent=int(sys.argv[1])
deadline=time.monotonic()+14400
while True:
    try:os.kill(parent,0)
    except ProcessLookupError:break
    if time.monotonic()>deadline:raise TimeoutError('Calibration process did not finish')
    time.sleep(1)
for cal in ('ultrachat','gsm8k-train'):
    if not (OUT/'calibration'/cal/'validation.json').exists():
        raise RuntimeError(f'Calibration failed: {cal}; see calibrate.log')
python=str(REPO/'.venv/bin/python')
for stage in ('smoke','measure','report'):
    subprocess.run([python,str(OUT/'experiment.py'),stage],cwd=REPO,check=True)
subprocess.run([python,str(OUT/'analyze.py')],cwd=REPO,check=True)
