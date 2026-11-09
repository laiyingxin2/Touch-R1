"""Touch-R1-inspired ordinal reward on the public dataset's actual JSON schema."""
import json
from score_quality_v3 import scales, schema_valid, relation_from_evidence
from score_quality_v2 import normalize

def parse_strict(text):
    def unique(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj: raise ValueError('Duplicate JSON key')
            obj[key] = value
        return obj
    try:
        obj = json.loads(text, object_pairs_hook=unique)
        return obj if isinstance(obj,dict) and set(obj)=={'answer','evidence'} else None
    except (ValueError,TypeError): return None

def reward(row, text):
    pred = parse_strict(text)
    if pred is None: return {'total':-0.1, 'exact_answer':False, 'ordinal':0., 'schema':False}
    gold = json.loads(row['messages'][-1]['content']); ranks = scales(row)
    exact = normalize(pred.get('answer')) == normalize(gold['answer'])
    try: valid = bool(schema_valid(row,pred))
    except (KeyError,TypeError,ValueError): valid = False
    # A missing or malformed field earns zero, never the neighbor reward.
    def one(prop, value, target):
        names = ranks[prop]
        if not isinstance(value,str) or value not in names: return 0.
        d = abs(names.index(value)-names.index(target))
        return 1. if d==0 else 0.4 if d==1 else 0.
    terms = []
    if row['task']=='property_description':
        a=pred.get('answer'); a=a if isinstance(a,dict) else {}
        terms=[one(p,a.get(p),v) for p,v in gold['answer'].items()]
    else:
        ev=pred.get('evidence'); ev=ev if isinstance(ev,dict) else {}
        for letter, attrs in gold['evidence'].items():
            got=ev.get(letter); got=got if isinstance(got,dict) else {}
            terms.extend(one(p,got.get(p),v) for p,v in attrs.items())
    ordinal=sum(terms)/len(terms) if terms else 0.
    try: derived=relation_from_evidence(row,pred)
    except (KeyError,TypeError,ValueError): derived=None
    consistent = derived is not None and normalize(pred.get('answer'))==normalize(derived)
    # Consistency alone cannot compensate for incorrect physical evidence.
    consistency = float(consistent and exact)
    return {'total':0.5*exact+0.35*ordinal+0.1*valid+0.05*consistency,
            'exact_answer':exact, 'ordinal':ordinal, 'schema':valid,
            'correct_relation_consistency':bool(consistency), 'cross_sensor_reward':0.}
