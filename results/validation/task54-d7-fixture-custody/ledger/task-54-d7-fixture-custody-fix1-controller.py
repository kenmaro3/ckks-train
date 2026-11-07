"""Exclusive fresh-attempt custody; propagate the actual child exit status."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ledger=Path(__file__).resolve().parent
root=ledger.parents[2]
prefix='task-54-d7-fixture-custody-fix1-attempt-01'
attempt_id='task54-fixture-custody-fix1-attempt-01'
def pin(path):
    data=path.read_bytes()
    return dict(path=str(path),byte_length=len(data),sha256=hashlib.sha256(data).hexdigest())
inventory_path=ledger/'task-54-d7-fixture-custody-fix1-test-inventory.json'
inventory=json.loads(inventory_path.read_text())
driver=ledger/'task-54-d7-fixture-custody-fix1-driver.py'
assert pin(driver)==inventory['driver']
assert pin(Path(__file__).resolve())==inventory['controller']
output_path=ledger/(prefix+'-result.json')
stdout_path=ledger/(prefix+'-stdout.log')
stderr_path=ledger/(prefix+'-stderr.log')
outcome_path=ledger/(prefix+'-process-outcome.json')
command_path=ledger/(prefix+'-command.json')
assert not output_path.exists()
command=[sys.executable,'-B',str(driver),'--attempt-id',attempt_id,'--output',str(output_path)]
metadata=dict(attempt_id=attempt_id,command=command,cwd=str(root),driver=pin(driver),controller=pin(Path(__file__).resolve()),inventory=pin(inventory_path),python=pin(Path(sys.executable)),output_paths=[str(p) for p in (output_path,stdout_path,stderr_path,outcome_path)])
with command_path.open('x') as command_file, stdout_path.open('xb') as stdout, stderr_path.open('xb') as stderr, outcome_path.open('x') as outcome:
    json.dump(metadata,command_file,indent=2); command_file.write('\n'); command_file.flush()
    started=datetime.now(timezone.utc).isoformat()
    completed=subprocess.run(command,cwd=root,stdout=stdout,stderr=stderr)
    finished=datetime.now(timezone.utc).isoformat()
    stdout.flush(); stderr.flush()
    result=dict(attempt_id=attempt_id,command=command,cwd=str(root),driver=pin(driver),command_metadata=pin(command_path),started_utc=started,finished_utc=finished,actual_exit_status=completed.returncode,stdout=pin(stdout_path),stderr=pin(stderr_path),result=pin(output_path) if output_path.exists() else None,exit_status_propagated=True)
    json.dump(result,outcome,indent=2); outcome.write('\n'); outcome.flush()
print(json.dumps(result,sort_keys=True))
raise SystemExit(completed.returncode)
