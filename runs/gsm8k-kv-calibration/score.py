"""Score stored generations using the repository-pinned EvalScope GSM8K rules."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

from evalscope.metrics.math_parser import extract_answer
from evalscope.metrics.metric import Accuracy

OUT=Path(__file__).resolve().parent
REVISION='acd09b44384d53174768bb1063f675420f76fae9'

def main():
    direct=json.loads(importlib.metadata.distribution('evalscope').read_text('direct_url.json'))
    assert direct['vcs_info']['commit_id']==REVISION
    directory,dataset=map(Path,sys.argv[1:3])
    read=lambda p:[json.loads(line) for line in p.read_text().splitlines()]
    inputs=read(dataset)
    answers=read(directory/f'run_{dataset.stem}_threshold_0.95.jsonl')
    gold={r['id']:r for r in read(OUT/'data/gold.jsonl')}
    assert [r['id'] for r in inputs]==[r['id'] for r in answers]
    scorer=Accuracy(numeric=True)
    examples=[('\\boxed{42}','42',1),('The final answer is 17.','17',1),('\\boxed{1,200}','1200',1),('\\boxed{0}','1',0),('No answer','5',0)]
    for text,target,expected in examples:
        assert scorer.apply([extract_answer(text)],[target])[0]==expected
    predictions=[]
    for row in answers:
        extracted=extract_answer(row['answer'])
        target=gold[row['id']]['target']
        correct=scorer.apply([extracted],[target])[0]
        predictions.append(dict(id=row['id'],prediction=extracted,target=target,correct=bool(correct),
                                generated_length=row['generated_length'],answer=row['answer']))
    correct=sum(r['correct'] for r in predictions)
    n=len(predictions)
    p=correct/n
    z=1.959963984540054
    center=(p+z*z/(2*n))/(1+z*z/n)
    half=z*((p*(1-p)/n+z*z/(4*n*n))**0.5)/(1+z*z/n)
    payload=dict(accuracy=p,correct=correct,samples=n,accuracy_wilson95=[center-half,center+half],
                 extraction_failures=sum(not r['prediction'] for r in predictions),evalscope_commit=REVISION,
                 evaluator='GSM8KAdapter.extract_answer -> Accuracy(numeric=True); offline generated answers',
                 gold_sha256=hashlib.sha256((OUT/'data/gold.jsonl').read_bytes()).hexdigest())
    (directory/'accuracy.json').write_text(json.dumps(payload,indent=2)+'\n')
    (directory/'predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in predictions))
    print(json.dumps(payload,indent=2))

if __name__=='__main__':main()
