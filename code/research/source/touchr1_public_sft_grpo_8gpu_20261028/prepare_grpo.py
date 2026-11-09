"""Pin v13 RL prompts and public replay, excluding test/validation supervision."""
import collections
from concurrent.futures import ThreadPoolExecutor
import gzip
import importlib.util
import json
from pathlib import Path
import shutil
import sqlite3
from common import ROOT,V12,PARENT,read,write,digest,emit
import data as source_data

OLD=Path('/mnt/workspace/lyx/touchr1_continual_dual_20261028')
SOURCE=Path('/mnt/workspace/lyx/touchreason_full_review_20261023_v1/paper_aligned_quick_v13')

def rows(path):
    with (gzip.open(path,'rt') if str(path).endswith('.gz') else Path(path).open()) as f:
        for s in f:
            if s.strip():yield json.loads(s)

def main():
    if (ROOT/'data_manifest.json').exists():return
    manifest=read(SOURCE/'manifest.json');src=SOURCE/'grpo_qa.candidates.jsonl.gz'
    assert digest(src)==manifest['quota_files']['grpo_qa']['sha256']
    spec=importlib.util.spec_from_file_location('v13_rl_loader',SOURCE/'quick_candidate_loader.py')
    loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
    source_data.SOURCE=SOURCE
    clips={r['clip_id']:r for r in rows(SOURCE/'clips.locators.jsonl.gz')}
    episodes={r['episode_id']:r for r in rows(SOURCE/'episodes.jsonl')}
    engine_eps=read(V12/'episodes.json')
    protected={r['group_id'] for r in rows(SOURCE/'source_objects.jsonl') if r['split'] in ['val','test','heldout_related']}
    aliases=read(V12/'verified_source_aliases.json')
    extra_aliases=ROOT/'verified_grpo_source_aliases.json'
    if extra_aliases.exists():aliases.update(read(extra_aliases))
    rl_eps={k:v for k,v in episodes.items() if v['stage']=='grpo'}
    def accessible(pair):
        eid,ep=pair;p=ep['source']['path'];return eid,Path(aliases.get(p,{}).get('path',p)).exists()
    access=dict(ThreadPoolExecutor(max_workers=24).map(accessible,rl_eps.items()))
    write(ROOT/'media_availability.json',dict(episodes=len(access),accessible=sum(access.values()),
        missing_episode_ids=[k for k,v in access.items() if not v],
        unverified_relocation_not_silently_used=True))
    temp=Path('/dev/shm/lyx_public_sft_GRPO_20261028.sqlite3');assert not temp.exists()
    con=sqlite3.connect(temp)
    con.execute('CREATE TABLE examples(idx INTEGER PRIMARY KEY,id TEXT UNIQUE,record TEXT NOT NULL)')
    pools=collections.defaultdict(list);tasks=collections.Counter();excluded=collections.Counter();image_checked=set()
    def insert(row,pool):
        cur=con.execute('INSERT INTO examples(id,record) VALUES(?,?)',(row['id'],json.dumps(row,ensure_ascii=False)))
        pools[pool].append(cur.lastrowid)
    excluded_file=ROOT/'excluded_media_rows.jsonl'
    with excluded_file.open('w') as failures:
        for n,raw in enumerate(rows(src),1):
            assert raw['split']=='train' and raw['stage']=='grpo'
            for eid in raw['episode_ids']:
                assert eid in engine_eps and episodes[eid]['split']=='train' and episodes[eid]['group_id'] not in protected
            row=source_data.normalize_touchreason(raw,clips,loader);row['dataset']='touchreason_v13'
            bad=[m['episode_id'] for m in row['media'] if m['kind']=='raw' and not access[m['episode_id']]]
            if bad:
                failures.write(json.dumps(dict(id=row['id'],episode_ids=bad,reason='source missing; relocation identity not verified'))+'\n')
                excluded[row['task']]+=1
            else:
                for media in row['media']:
                    if media['kind']=='images':
                        for i,p in enumerate(media['paths']):
                            if p not in image_checked:
                                assert Path(p).is_file(),p
                                if media.get('sha256'):assert digest(p)==media['sha256'][i]
                                image_checked.add(p)
                insert(row,'native_grpo');tasks[row['task']]+=1
            if n%10000==0:
                con.commit();emit(ROOT,'normalize_grpo',source_n=n,usable_n=len(pools['native_grpo']),excluded_n=sum(excluded.values()))
    assert n==120000
    old=sqlite3.connect('file:'+read(OLD/'data_manifest.json')['runtime_database']+'?mode=ro&immutable=1',uri=True)
    old_splits=read(OLD/'splits.json');mapping={}
    for pool in ['public_train','native_val','public_val']:
        for idx in old_splits[pool]:
            row=json.loads(old.execute('SELECT record FROM examples WHERE idx=?',(idx,)).fetchone()[0])
            if pool=='public_train':row=dict(row,stage='grpo',origin_stage='public_sft_replay')
            insert(row,pool);mapping[idx]=pools[pool][-1]
    con.commit();assert con.execute('PRAGMA integrity_check').fetchone()[0]=='ok';con.close()
    disk=ROOT/'train_val.sqlite3';shutil.copyfile(temp,disk)
    assert digest(disk)==digest(temp)
    splits=dict(pools,screen_validation=[mapping[i] for i in old_splits['screen_validation']],
        full_validation=[mapping[i] for i in old_splits['full_validation']])
    write(ROOT/'splits.json',splits)
    write(ROOT/'data_manifest.json',dict(database=str(disk),runtime_database=str(temp),
        database_bytes=disk.stat().st_size,database_sha256=digest(disk),
        source_v13=str(src),source_sha256=digest(src),source_rows=120000,
        pool_counts={k:len(v) for k,v in pools.items()},excluded_missing_media=sum(excluded.values()),
        excluded_per_task=dict(excluded),test_in_training=False,
        validation_in_training=False,weak_supervision_experiment=True,
        cross_sensor_physical_pairs_verified=False,reviewed_images_verified=len(image_checked)))
    write(ROOT/'preflight_data.json',dict(passed=True,training_rows=len(pools['native_grpo'])+len(pools['public_train']),
        missing_media_rows_excluded_with_audit=sum(excluded.values()),test_used=False,
        public_train_val_media_separation_audit=str(OLD/'clean_preflight_audit.json')))
    emit(ROOT,'data_ready',pool_counts={k:len(v) for k,v in pools.items()},excluded=sum(excluded.values()))

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        write(ROOT/'prepare_failure.json',dict(error=str(exc),type=type(exc).__name__));raise
