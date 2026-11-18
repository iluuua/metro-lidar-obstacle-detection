"""Portable local validator; no datasets or GPU needed."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parents[1]
node=shutil.which('node')
commands=[[sys.executable,'-m','unittest','discover','-s','tests','-v']]
if node:
    commands += [[node,'--check','viewer/app.js'],
                 [node,'--test',*[str(p.relative_to(ROOT)) for p in sorted((ROOT/'tests').glob('*.test.js'))]]]
results=[]
for cmd in commands:
    done=subprocess.run(cmd,cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    print(done.stdout,end='')
    results.append({'command':cmd,'returncode':done.returncode,'output':done.stdout})
report={'utc':datetime.now(timezone.utc).isoformat(),'status':'PASS' if node and all(r['returncode']==0 for r in results) else 'FAIL',
        'checks':results,'missing':[] if node else ['Node.js required for viewer validation'],
        'not_validated':['ROS2/Docker runtime','detector ground truth metrics','physical realism of geometric curve warp']}
out=ROOT/'output';out.mkdir(parents=True,exist_ok=True)
(out/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
(out/'last_validator.txt').write_text('\n'.join(r['output'] for r in results))
print(report['status']);sys.exit(0 if report['status']=='PASS' else 1)
