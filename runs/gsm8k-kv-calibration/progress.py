import json
from pathlib import Path
import re

out=Path(__file__).resolve().parent
done={stage:sum(json.loads(p.read_text()).get('status')=='passed' for p in (out/stage).glob('*/result.json')) for stage in ('smoke','measurements')}
commands=list((out/'smoke').glob('*/command.json'))+list((out/'measurements').glob('*/command.json'))
current=max(commands,key=lambda p:json.loads(p.read_text())['started']) if commands else None
status={'complete':done}
if current:
    d=current.parent
    log=(d/'run.log').read_text(errors='replace') if (d/'run.log').exists() else ''
    matches=re.findall(r'\[Iter=\s*(\d+)\]nfe=\s*(\d+), Token number=\s*(\d+), Sample_time=([\d.]+)',log)
    total=1319 if d.parent.name=='measurements' else 16
    processed=min(int(matches[-1][0])+8,total) if matches else 0
    status.update(stage=d.parent.name,case=d.name,processed=processed,total=total,
                  generation_seconds_so_far=round(sum(float(m[3]) for m in matches),2),
                  tokens_so_far=sum(int(m[2]) for m in matches),
                  last_log_line=log.splitlines()[-1][:180] if log else '')
    if (d/'result.json').exists():status['case_status']=json.loads((d/'result.json').read_text())['status']
(out/'status.json').write_text(json.dumps(status,indent=2)+'\n')
print(json.dumps(status,ensure_ascii=False))
