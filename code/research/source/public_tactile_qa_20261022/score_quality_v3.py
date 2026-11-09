"""Score v3 answers, evidence correctness, and whether stated evidence supports the relation."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from score_quality_v2 import read, score as score_v2, normalize, response_object


def scales(row):
    text=row['messages'][0]['content'].split('Category scales, from low to high: ',1)[1].split('. Use these exact category names.',1)[0]
    return {p:labels.split(' < ') for p,labels in (item.split(' = ',1) for item in text.split('; '))}


def relation_from_evidence(row,prediction):
    if row['task'] not in ['property_comparison','property_selection']:return None
    prop=row['evaluation']['property'];names=scales(row)[prop];evidence=prediction.get('evidence',{})
    if not isinstance(evidence,dict):return None
    try:
        values=[names.index(evidence[k][prop]) for k in 'abc'[:len(row['media_ids'])]]
    except (KeyError,ValueError,TypeError):return None
    direction=row['evaluation']['direction']
    if row['task']=='property_comparison':
        delta=(values[0]-values[1])*direction
        return 'same_category' if delta==0 else 'yes' if delta>0 else 'no'
    target=(max if direction>0 else min)(values)
    return 'abc'[values.index(target)] if values.count(target)==1 else None


def schema_valid(row,prediction):
    answer=prediction.get('answer');evidence=prediction.get('evidence');task=row['task'];allowed=scales(row)
    if not isinstance(evidence,dict):return False
    if task=='property_description':
        return isinstance(answer,dict) and set(answer)==set(allowed) and all(answer[p] in choices for p,choices in allowed.items()) and evidence=={}
    if set(evidence)!=set('abc'[:len(row['media_ids'])]):return False
    if not all(isinstance(attrs,dict) and set(attrs)==set(allowed) and all(attrs[p] in choices for p,choices in allowed.items()) for attrs in evidence.values()):return False
    if task=='property_comparison':return isinstance(answer,str) and answer in ['yes','no','same_category']
    if task=='property_selection':return isinstance(answer,str) and answer in 'abc' and len(answer)==1
    return isinstance(answer,dict) and set(answer)==set('abc') and len(set(answer.values()))==3 and sorted(answer.values())==row['evaluation']['candidate_names']


def score(gold,predictions):
    result=score_v2(gold,predictions)
    by_id={p['id']:p for p in predictions}
    joint=valid=consistent=usable=reasoning=0
    confusion=defaultdict(Counter)
    for row in gold:
        target=json.loads(row['messages'][1]['content']);pred=response_object(by_id.get(row['id'],{}))
        good=normalize(pred.get('answer'))==normalize(target['answer'])
        joint+=good and normalize(pred.get('evidence',{}))==normalize(target['evidence'])
        try:valid+=schema_valid(row,pred)
        except (TypeError,ValueError):pass
        if row['task'] in ['property_comparison','property_selection']:
            reasoning+=1;derived=relation_from_evidence(row,pred);usable+=derived is not None
            consistent+=derived is not None and normalize(pred.get('answer'))==normalize(derived)
        if row['task']=='property_comparison':
            answer=pred.get('answer');label=answer if isinstance(answer,str) and answer in ['yes','no','same_category'] else 'invalid_or_missing'
            confusion[row['dataset']][(target['answer'],label)]+=1
    result.update(answer_and_exact_evidence_joint_accuracy=joint/len(gold) if gold else None,
                  output_schema_valid_accuracy=valid/len(gold) if gold else None,
                  relation_evidence_consistency_accuracy=consistent/reasoning if reasoning else None,
                  relation_cases_with_usable_evidence=usable,relation_cases=reasoning,
                  distinct_media_items=len({mid for row in gold for mid in row['media_ids']}),
                  distinct_source_objects=len({(row['dataset'],ob) for row in gold for ob in row['curation']['source_objects']}))
    result['comparison_confusion']={ds:{f'{truth}->{predicted}':n for (truth,predicted),n in sorted(bins.items())} for ds,bins in confusion.items()}
    recalls=[]
    for ds,bins in confusion.items():
        for truth in ['yes','no','same_category']:
            total=sum(n for (t,p),n in bins.items() if t==truth)
            if total:recalls.append(bins[(truth,truth)]/total)
    result['comparison_macro_recall_over_present_dataset_answer_classes']=sum(recalls)/len(recalls) if recalls else None
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gold',required=True);p.add_argument('--predictions',required=True);p.add_argument('--output');a=p.parse_args()
    result=score(read(a.gold),read(a.predictions));text=json.dumps(result,ensure_ascii=False,indent=2)
    if a.output:Path(a.output).write_text(text)
    print(text)
