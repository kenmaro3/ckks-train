"""External controller: validate statically, own one fresh child, retain evidence.

Root must verify this file before executing it in its separately reviewed base
interpreter/bootstrap. No target venv is imported or started at module import.
"""
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
import types

class PrelaunchError(RuntimeError):
    pass

def require(ok,message):
    if not ok: raise PrelaunchError(message)

def load_release(path,sha):
    """Independent bounded digest/duplicate-key check BEFORE shared code load."""
    p=Path(path)
    require(p.is_absolute() and str(p.resolve())==str(p),'noncanonical release')
    st=p.lstat(); require(stat.S_ISREG(st.st_mode) and st.st_size<=8*1024*1024,'release kind/size')
    raw=p.read_bytes(); require(len(raw)==st.st_size and hashlib.sha256(raw).hexdigest()==sha,'release pin mismatch')
    def pairs(items):
        result={}
        for k,v in items:
            require(k not in result,'duplicate release key'); result[k]=v
        return result
    s=json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda _: (_ for _ in ()).throw(PrelaunchError('nonfinite JSON')))
    require(type(s) is dict and type(s.get('programs')) is dict and set(s['programs'])=={'prelaunch','launcher'},'missing program bindings')
    item=s['programs']['launcher']; require(type(item) is dict and set(item)=={'path','sha256'},'bad launcher binding')
    file=Path(item['path']); require(file.is_absolute() and str(file.resolve())==str(file),'launcher noncanonical')
    fs=file.lstat(); require(stat.S_ISREG(fs.st_mode) and fs.st_size<=1024*1024,'launcher kind/size')
    source=file.read_bytes(); require(len(source)==fs.st_size and hashlib.sha256(source).hexdigest()==item['sha256'],'launcher pin mismatch')
    library=types.ModuleType('_task54_prelaunch_validation'); library.__file__=str(file)
    # Exactly the verified buffer; definitions only. No main or target startup.
    exec(compile(source,str(file),'exec',dont_inherit=True),library.__dict__)
    validated=library.load_release(path,sha)
    own=validated['programs']['prelaunch']
    require(str(Path(__file__).resolve())==own['path'],'wrong prelaunch identity')
    library.verify_file(__file__,own['sha256'])
    return validated,library

def validate_static(spec,library):
    return library.validate_static(spec)

def build_child_command(spec,library):
    return library.build_child_command(spec)

def run_child(command,env,cwd,record):
    """Only this Popen child is killed; preserve nonzero/timeout/interruption facts."""
    record['actual_argv']=list(command); record['environment']=dict(env); record['cwd']=cwd
    record['child_started_unix']=time.time(); record['success']=False
    child=subprocess.Popen(command,env=env,cwd=cwd,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    record['child_pid']=child.pid
    stdout=stderr=b''; collected=False; primary=None
    try:
        try:
            stdout,stderr=child.communicate(timeout=record['timeout_seconds']); collected=True
        except subprocess.TimeoutExpired as exc:
            record['timed_out']=True; stdout=exc.output or b''; stderr=exc.stderr or b''
    except BaseException as exc:
        primary=exc; record['controller_error']={'type':type(exc).__name__,'message':str(exc)}
    finally:
        errors=[]
        if not collected:
            try:
                if child.poll() is None: child.kill()
            except BaseException as exc: errors.append({'operation':'kill','type':type(exc).__name__,'message':str(exc)})
            try:
                stdout,stderr=child.communicate(timeout=record['cleanup_seconds']); collected=True
            except BaseException as exc:
                errors.append({'operation':'collect','type':type(exc).__name__,'message':str(exc)})
                if isinstance(exc,subprocess.TimeoutExpired): stdout=exc.output or stdout; stderr=exc.stderr or stderr
        try: terminal=child.poll() is not None
        except BaseException as exc:
            terminal=False; errors.append({'operation':'poll','type':type(exc).__name__,'message':str(exc)})
        record.update(child_finished_unix=time.time(),child_returncode=child.returncode,child_terminal_observed=terminal,child_streams_complete=collected,cleanup_errors=errors,child_stdout_hex=stdout.hex(),child_stderr_hex=stderr.hex(),child_stdout_sha256=hashlib.sha256(stdout).hexdigest(),child_stderr_sha256=hashlib.sha256(stderr).hexdigest(),child_cleanup_status='terminal_and_streams_collected' if terminal and collected else 'unresolved_terminal_or_streams')
    if primary is not None: raise primary
    require(record['child_terminal_observed'] and record['child_streams_complete'],'unresolved child terminal/streams')
    require(not record.get('timed_out'),'child timeout')
    require(child.returncode==0,'child nonzero')
    record['success']=True

def main(argv=None):
    args=sys.argv[1:] if argv is None else argv
    record={'kind':'task54_prelaunch_operational_evidence','status':'failed','success':False,'started_unix':time.time(), 'controller': {'executable':sys.executable,'resolved_executable':str(Path(sys.executable).resolve()),'argv':list(sys.orig_argv),'prefix':sys.prefix,'base_prefix':sys.base_prefix,'isolated':sys.flags.isolated,'dont_write_bytecode':sys.dont_write_bytecode,'no_site':sys.flags.no_site,'nice':os.nice(0)}}
    spec=library=None
    try:
        require(len(args)==4 and args[0]=='--release' and args[2]=='--release-sha256','exact transport required')
        spec,library=load_release(args[1],args[3]); record['release']={'path':args[1],'sha256':args[3]}
        record['static']=validate_static(spec,library)
        record['timeout_seconds']=spec['timeout_seconds']; record['cleanup_seconds']=spec['cleanup_seconds']
        command=build_child_command(spec,library)
        run_child(command,spec['environment'],spec['cwd'],record)
        child_record=library.strict_json(library.read_regular(spec['evidence']['child']))
        expected_status={'startup-only':'observation_only_not_runtime_approval','context':'context_observed_not_capture_approval','capture':'exporter_main_returned_zero_not_scientific_claim'}[spec['purpose']]
        require(child_record.get('status')==expected_status and child_record.get('purpose')==spec['purpose'] and child_record.get('release')==record['release'],'missing/mismatched child terminal evidence')
        record['child_evidence_sha256']=hashlib.sha256(library.read_regular(spec['evidence']['child'])).hexdigest()
        record['status']='child_terminal_zero_collected'; return 0
    except BaseException as exc:
        record.update(status='failed_no_retry',success=False,error_type=type(exc).__name__,error=str(exc)); print(json.dumps(record),file=sys.stderr); return 1
    finally:
        record['finished_unix']=time.time()
        if spec is not None: library.write_new(spec['evidence']['parent'],record)

if __name__=='__main__':
    raise SystemExit(main())
