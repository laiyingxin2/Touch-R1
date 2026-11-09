"""Frozen native TouchReason reconstructed-benchmark scorer, extracted verbatim.

Source: touchreason_bench4800_v2_20261023/common.py. Importing only scoring
avoids installing its separate vision monkeypatch into the trained encoder.
"""
import collections
import re
import numpy as np
from scipy.stats import spearmanr

def parse_result(r,text):
 text=text.strip();task=r['task'].replace('joint_partial_','joint_');pred=None;correct=False;value=0.;valid=False
 if task in ['property','joint_mc','pair','best']:
  m=re.fullmatch(r'([A-Z])[.)]?',text.upper());pred=m.group(1) if m else None
  valid=pred in r['options'];correct=valid and pred==r['answer'];value=float(correct)
 elif task=='ordinal':
  valid=bool(re.fullmatch('[1-3]',text));pred=int(text) if valid else None
  value=abs(pred-r['gold']['rank']) if valid else 2;correct=valid and value==0
 elif task=='joint_em':
  pred=text.lower().strip('` .\n').split('|');valid=len(pred)==len(r['gold'])
  correct=valid and '|'.join(x.strip() for x in pred)==r['answer'];value=float(correct)
 elif task=='joint_ms':
  pred=[x for x in re.split(r'[,\s]+',text.upper()) if x];valid=bool(pred) and len(set(pred))==len(pred) and all(x in r['options'] for x in pred)
  gold=set(r['answer'].split(','));ps=set(pred);value=2*len(gold&ps)/(len(gold)+len(ps)) if valid else 0.;correct=valid and gold==ps
 elif task=='ranking':
  valid=bool(re.fullmatch(r'[1-3]\s*\|\s*[1-3]\s*\|\s*[1-3]\s*\|\s*[1-3]',text))
  pred=[int(x) for x in text.split('|')] if valid else None
  value=float(spearmanr(r['gold']['ranks'],pred).statistic) if valid and len(set(pred))>1 else 0.
  if not np.isfinite(value):value=0.
  correct=valid and value==1.
 out={'qa_id':r['qa_id'],'task':r['task'],'attribute':r['attribute'],'anchor_object_uid':r['anchor_object_uid'],
  'sample_ids':r['sample_ids'],'output':text,'prediction':pred,'answer':r['answer'],'valid':valid,'correct':bool(correct),'value':value,'gold':r['gold']}
 if task=='property':out['predicted_label']=r['options'].get(pred)
 return out

def summarize(records):
 tasks={}
 for task in sorted({r['task'] for r in records}):
  rs=[r for r in records if r['task']==task]
  tasks[task]={'n':len(rs),'score':sum(r['value'] for r in rs)/len(rs),'invalid_n':sum(not r['valid'] for r in rs)}
 attrs={}
 for a in ['hardness','roughness','protrusion','material']:
  rs=[r for r in records if r['task']=='property' and r['attribute']==a]
  if not rs:continue
  counts=collections.Counter(r['gold']['label'] for r in rs);cor=collections.Counter(r['gold']['label'] for r in rs if r['correct'])
  recalls={g:cor[g]/n for g,n in counts.items()};attrs[a]={'n':len(rs),'accuracy':sum(r['correct'] for r in rs)/len(rs),
   'balanced_accuracy':sum(recalls.values())/len(recalls),'class_recall':recalls,'counts':dict(counts)}
 return {'tasks':tasks,'attributes':attrs,'mean_property_balanced_accuracy':sum(v['balanced_accuracy'] for v in attrs.values())/len(attrs) if attrs else None,
  'official_paper_avg':None,'caveat':'Local reconstructed instances; equal question count does not establish official benchmark equivalence.'}
