"""Read-only current original-producer inventory; stdlib only, no scientific imports."""
from pathlib import Path
import hashlib,json,os,sys,time,importlib.metadata
root=Path('/home/kmihara/reboot')
site=root/'venv/lib/python3.11/site-packages'
assert root.is_dir() and site.is_dir()
def file_info(path):
    path=Path(path)
    if not path.is_file():return {'path':str(path),'exists':False}
    raw=path.read_bytes();s=path.stat()
    return {'path':str(path),'resolved':str(path.resolve()),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'mtime_ns':s.st_mtime_ns}
packages={}
selected={'numpy','torch','torchvision','pillow','scipy','seaborn','matplotlib'}
for d in importlib.metadata.distributions(path=[str(site)]):
    name=d.metadata['Name'].lower()
    if name not in selected:continue
    assert name not in packages
    info={'name':d.metadata['Name'],'version':d.version,'metadata':{}}
    for f in d.files or []:
        if f.name in ('METADATA','WHEEL','RECORD') and '.dist-info/' in str(f):
            info['metadata'][f.name]=file_info(d.locate_file(f))
    packages[name]=info
sources={}
for rel in ['src/ReBoot/lib/utils/data.py','src/ReBoot/experiments/4_Scalability/precision_analysis.py','reboot_full_setting.py','reboot_emlp23.py','venv/pyvenv.cfg']:
    sources[rel]=file_info(root/rel)
for rel in ['torchvision/transforms/transforms.py','torchvision/transforms/functional.py','torchvision/transforms/_functional_tensor.py','torch/version.py']:
    sources[rel]=file_info(site/rel)
# Large native files are only listed/stat'ed in this bounded read-only observation.
# They are not hash-attested or approved by this inventory.
native=[]
for base in [site/'torch/lib',root/'install/lib']:
    for p in sorted(base.glob('*.so*')):
        if p.is_file():
            s=p.stat();native.append({'path':str(p),'resolved':str(p.resolve()),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'sha256':'not_measured'})
log=Path('/home/kmihara/hekit-runs/logs/R-41-FULL-2.log')
with log.open('rb') as f:
    f.seek(max(0,log.stat().st_size-8192));tail=f.read().decode(errors='replace').splitlines()[-4:]
print(json.dumps({'kind':'read_only_current_environment_inventory_not_historical_attestation_or_runtime_approval','observed_unix':time.time(),'observer_python':sys.version,'owner_uid':os.getuid(),'packages':packages,'sources':sources,'native_stat_only':native,'full2_log_tail':tail,'scientific_modules_loaded':[n for n in sys.modules if n.split('.')[0] in selected or n=='lib'],'full2_final_result_exists':Path('/home/kmihara/hekit-runs/out/R-41-FULL-2/result.json').exists()},indent=2))
