from pathlib import Path
import json,os,stat,time

root=Path('/home/kmihara/hekit-runs/logs')
rows=[]
for job in ('R-48-4','R-52-1','R-52-2'):
    path=root/(job+'.log')
    row={'job':job,'path':str(path),'kind':'bounded_nonterminal_log_observation'}
    fd=None
    try:
        fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
        before=os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_uid!=os.getuid():
            raise ValueError('refusing nonregular or nonowned log')
        offset=max(0,before.st_size-8192)
        raw=os.pread(fd,8192,offset)
        after=os.fstat(fd)
        row.update(exists=True,uid=before.st_uid,device=before.st_dev,inode=before.st_ino,
                   before_size=before.st_size,after_size=after.st_size,
                   before_mtime_ns=before.st_mtime_ns,after_mtime_ns=after.st_mtime_ns,
                   offset=offset,bytes_read=len(raw),tail=raw.decode(errors='replace').splitlines()[-12:],
                   concurrent_append_possible=True,scientific_acceptance=False)
    except FileNotFoundError:
        row.update(exists=False)
    except (OSError,ValueError) as error:
        row.update(refused=True,error_type=type(error).__name__,error=str(error))
    finally:
        if fd is not None:os.close(fd)
    rows.append(row)
print(json.dumps({'kind':'read_only_progress_not_scientific_result','observed_unix':time.time(),
                  'uid':os.getuid(),'nice':os.nice(0),'logs':rows},indent=2))
