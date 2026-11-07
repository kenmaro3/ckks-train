from pathlib import Path
import subprocess,sys,time
root=Path(__file__).parent
label=sys.argv[1]; cmd=sys.argv[2:]
with (root/(label+'.log')).open('x') as log:
 log.write('COMMAND '+repr(cmd)+'\n'); log.flush()
 p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
 log.write('\nEXIT '+str(p.returncode)+'\n')
print((root/(label+'.log')).read_text()[-6500:])
print('IMMUTABLE_LOG',root/(label+'.log'))
sys.exit(p.returncode)
