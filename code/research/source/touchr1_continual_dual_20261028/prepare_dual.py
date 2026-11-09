"""Audit the clean branch; freeze an explicitly test-contaminated fitting branch."""
import json
import re
import shutil
import sqlite3
from pathlib import Path
from common import ROOT, V12, PARENT, PUBLIC, read, write, digest
import importlib.util
_spec=importlib.util.spec_from_file_location('dual_prepare',ROOT/'prepare.py')
_prepare=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_prepare)
stratified=_prepare.stratified
from data import target

def rows(path):
    with Path(path).open() as f:
        return [json.loads(s) for s in f if s.strip()]

def audit():
    splits=read(ROOT/'splits.json')
    con=sqlite3.connect('file:'+str(read(ROOT/'data_manifest.json')['runtime_database'])+'?mode=ro&immutable=1',uri=True)
    episodes=read(V12/'episodes.json')
    protected=[r for r in rows(PARENT/'snapshot/source_objects.jsonl') if r['split'] in ['val','test','heldout_related']]
    protected_samples={r['sample_id'] for r in protected}
    protected_groups={r['group_id'] for r in protected}
    refs=0
    print('Audit replay episode/group identities',flush=True)
    for idx in splits['native_replay']:
        r=json.loads(con.execute('SELECT record FROM examples WHERE idx=?',(idx,)).fetchone()[0])
        assert r['episode_ids'], 'Replay identity audit needs episode IDs'
        for eid in r['episode_ids']:
            ep=episodes[eid]; refs+=1
            assert ep['split']=='train' and ep['sample_id'] not in protected_samples
            assert ep['group_id'] not in protected_groups
    print('Replay identities passed',refs,flush=True)
    memberships={}
    for split in ['train','val','test']:
        path=PUBLIC/'organized/annotations/quality_v3'/(split+'.jsonl')
        # Extract input media membership only, without importing test targets.
        memberships[split]=set()
        with path.open() as f:
            for line in f:
                match=re.search(r'"media_ids"\s*:\s*(\[[^\]]*\])',line)
                assert match
                memberships[split].update(json.loads(match.group(1)))
    assert not memberships['train'] & (memberships['val']|memberships['test'])
    media={r['id']:r for r in rows(PUBLIC/'organized/media_index.jsonl')}
    missing=[]
    print('Audit public source paths',flush=True)
    for mid in memberships['train']|memberships['val']:
        m=media[mid]; path=PUBLIC/'organized'/m['path']
        if not path.exists():missing.append(str(path))
    assert not missing, missing[:5]
    value=dict(passed=True, replay_episode_references_checked=refs,
        protected_sample_overlap=0,protected_group_overlap=0,
        public_unique_media_counts={k:len(v) for k,v in memberships.items()},
        public_train_val_media_overlap=0,public_train_test_media_overlap=0,
        public_source_paths_missing=0,physical_identity_independently_verified=False,
        supersedes_vacuous_test_media_index_check=True)
    write(ROOT/'clean_preflight_audit.json',value)
    meta=read(ROOT/'data_manifest.json')
    meta['public_test_media_index_n']=len(memberships['test'])
    meta['substantive_audit']=str(ROOT/'clean_preflight_audit.json')
    write(ROOT/'data_manifest.json',meta)
    print('CLEAN AUDIT PASSED',flush=True)

def fit_data():
    if (ROOT/'test_fit_dataset_manifest.json').exists():return
    folder=ROOT/'test_fit_data';folder.mkdir(exist_ok=True)
    gold_src=PARENT/'snapshot/test_qa.jsonl'
    shutil.copyfile(gold_src,folder/'gold.jsonl')
    gold={r['qa_id']:r for r in rows(gold_src)}
    inputs_path=V12/'test_inputs/touchreason.jsonl'
    inputs=rows(inputs_path)
    assert len(inputs)==len(gold)==4800
    path=folder/'test_fit.sqlite3';assert not path.exists()
    runtime=Path('/dev/shm/lyx_touchr1_TEST_FIT_20261028.sqlite3')
    assert not runtime.exists()
    con=sqlite3.connect(runtime)
    con.execute('CREATE TABLE examples(idx INTEGER PRIMARY KEY,id TEXT UNIQUE,record TEXT NOT NULL)')
    by_id={}
    for r in inputs:
        assert r['native_id'] in gold and 'target' not in r
        r=dict(r,split='test_fit',stage='sft',answer=gold[r['native_id']]['answer'],
            target=target(str(gold[r['native_id']]['answer'])),
            test_used_for_training=True,same_data_train_and_eval=True)
        cur=con.execute('INSERT INTO examples(id,record) VALUES(?,?)',(r['id'],json.dumps(r,ensure_ascii=False)))
        by_id[r['id']]=cur.lastrowid
    con.commit();assert con.execute('PRAGMA integrity_check').fetchone()[0]=='ok';con.close()
    selected=stratified(inputs,128,lambda r:(r['task'],r.get('attribute')),'fit-screen-20261028')
    split=dict(fit_screen=[by_id[r['id']] for r in selected],fit_full=list(by_id.values()))
    write(ROOT/'test_fit_splits.json',split)
    shutil.copyfile(runtime,path)
    assert digest(runtime)==digest(path)
    write(ROOT/'test_fit_dataset_manifest.json',dict(database=str(path),runtime_database=str(runtime),
        database_bytes=path.stat().st_size,database_sha256=digest(path),n=4800,
        original_input_sha256=digest(inputs_path),gold_sha256=digest(gold_src),
        test_used_for_training=True,test_used_for_selection=True,same_data_train_and_eval=True,
        generalization_benchmark_result=False,official_reproduction=False,
        objective='Resubstitution fitting diagnostic; original test files unchanged.',
        no_fabricated_rationales=True))
    print('TEST FIT FROZEN',flush=True)

if __name__=='__main__':
    if not (ROOT/'clean_preflight_audit.json').exists():audit()
    fit_data()
