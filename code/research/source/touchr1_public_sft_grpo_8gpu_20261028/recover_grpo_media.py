"""Verify every relocated GRPO recording against original measured fingerprints."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import datetime
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
from common import ROOT,V12,read,write,emit,digest
import cv2

REF=Path('/mnt/workspace/lyx/touchreason_frame_annotation_20261023/frame_qc')
SOURCE=Path('/mnt/workspace/lyx/touchreason_full_review_20261023_v1/paper_aligned_quick_v13')
spec=importlib.util.spec_from_file_location('previous_video_verifier',V12/'verify_data0417_relocated.py')
verifier=importlib.util.module_from_spec(spec);spec.loader.exec_module(verifier)
cv2.setNumThreads(1)

def verify(candidate,episode):
    if candidate['source_kind']=='video':return verifier.verify(episode)
    result=dict(episode_id=episode['episode_id'],original_path=candidate['original_path'],
        path=candidate['resolved_path'],passed=False,matched_frames=0)
    hashes={}
    try:
        with gzip.open(REF/(episode['episode_id']+'.jsonl.gz'),'rt') as f:
            refs=[json.loads(s) for s in f]
        filenames=episode['source']['filenames'];assert len(refs)==len(filenames)
        for index,(ref,name) in enumerate(zip(refs,filenames)):
            assert ref['frame_index']==index
            p=Path(candidate['resolved_path'])/name
            before=digest(p);frame=cv2.imread(str(p),cv2.IMREAD_COLOR)
            assert frame is not None,(index,str(p))
            actual=hashlib.sha256(cv2.resize(frame,(64,64),interpolation=cv2.INTER_AREA).tobytes()).hexdigest()
            assert actual==ref['observed']['thumbnail_sha256'],('fingerprint mismatch',index,str(p))
            assert digest(p)==before,'Image changed during verification'
            hashes[name]=before;result['matched_frames']+=1
        result.update(passed=True,frame_sha256=hashes,
            verification='all original 64x64 BGR INTER_AREA SHA256 frame fingerprints')
    except Exception as exc:result['error']=str(exc)
    return result

def main():
    candidates=read(ROOT/'grpo_relocation_candidates.json');assert len(candidates)==69
    episodes={x['episode_id']:x for x in map(json.loads,(SOURCE/'episodes.jsonl').read_text().splitlines())}
    for c in candidates:
        assert Path(c['resolved_path']).exists(),c['resolved_path']
        assert (REF/(c['episode_id']+'.jsonl.gz')).is_file(),c['episode_id']
    results=[]
    with ThreadPoolExecutor(max_workers=12) as pool:
        fs=[pool.submit(verify,c,episodes[c['episode_id']]) for c in candidates]
        for f in as_completed(fs):
            value=f.result();results.append(value)
            emit(ROOT,'media_identity_verification',episode=value['episode_id'],passed=value['passed'],
                matched_frames=value['matched_frames'],checked=len(results),total=len(candidates),error=value.get('error'))
            write(ROOT/'grpo_media_recovery_progress.json',dict(checked=len(results),total=len(candidates),
                passed=sum(x['passed'] for x in results),matched_frames=sum(x['matched_frames'] for x in results),
                failures=[{k:v for k,v in r.items() if k!='frame_sha256'} for r in results if not r['passed']]))
    report=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),complete=True,
        total=len(results),passed=sum(x['passed'] for x in results),
        matched_frames=sum(x['matched_frames'] for x in results),results=results,
        original_data_modified=False,labels_modified=False,
        verification='Historical 64x64 BGR INTER_AREA frame SHA256; current full-file hashes also pinned.')
    write(ROOT/'grpo_media_recovery_verified.json',report)
    assert all(r['passed'] for r in results),'Verification failed; do not publish or train'
    aliases={}
    for r in results:
        keep=['path','episode_id','verification']
        keep+=['video_sha256','verified_frame_count'] if 'video_sha256' in r else ['frame_sha256']
        aliases[r['original_path']]={k:r[k] for k in keep}
    write(ROOT/'verified_grpo_source_aliases.json',aliases)
    emit(ROOT,'all_grpo_relocations_verified',episodes=len(aliases),frames=report['matched_frames'])

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        write(ROOT/'media_recovery_failure.json',dict(error=str(exc),type=type(exc).__name__));raise
