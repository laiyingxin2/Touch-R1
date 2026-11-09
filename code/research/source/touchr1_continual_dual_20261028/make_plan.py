"""Queue independent clean-validation and explicitly contaminated fitting studies."""
import copy
import random
from common import ROOT, read, write

def main():
    assert read(ROOT/'clean_preflight_audit.json')['passed']
    assert (ROOT/'test_fit_dataset_manifest.json').exists()
    if (ROOT/'queue_plan.json').exists():return
    splits=read(ROOT/'splits.json');fit=read(ROOT/'test_fit_splits.json')
    base=dict(seed=20261028,lr=1e-5,adapter_ratio=1,dropout=.05,weight_decay=.01,
        batch=64,warmup_fraction=.03,schedule='cosine',beta2=.95,gradient_clip=1.)
    parent=str(ROOT/'fast_experiments/initial_checkpoint')
    main_job=dict(folder=str(ROOT/'continual_8gpu/run'),config=base,
        initial_checkpoint=str(ROOT/'continual_8gpu/initial_checkpoint'),
        presentations=4*len(splits['main_epoch']),training_indices=splits['main_epoch'],
        validation_suite='full_validation',data_family='clean',microbatch=2,max_microbatch=8,
        checkpoint_presentations=4096,gpu_feature_cache=True,intermediate_validation=True)
    write(ROOT/'main_job.json',main_job)
    jobs=[]
    for family in ['test_fit','clean']:
        baseline=dict(folder=str(ROOT/'fast_experiments'/family/'parent_baseline'),config=base,
            initial_checkpoint=parent,presentations=0,training_indices=[],
            validation_suite='fit_screen' if family=='test_fit' else 'screen_validation',
            data_family=family,microbatch=1,max_microbatch=1,gpu_feature_cache=False)
        jobs.append(baseline)
    rng=random.Random(20261028)
    options=dict(lr=[3e-6,1e-5,3e-5,1e-4,3e-4],adapter_ratio=[1,2,4,8],
        batch=[8,16,32,64],dropout=[0.,.05,.1],weight_decay=[0.,.01,.1],
        beta2=[.95,.99,.999],gradient_clip=[.5,1.,2.],
        warmup_fraction=[0.,.03,.1],schedule=['cosine','linear','constant'])
    for index in range(32):
        for family in ['test_fit','clean']:
            config=dict(base)
            if index:
                config.update({k:rng.choice(v) for k,v in options.items()})
            # Eight regular LR/ratio anchors plus diverse random interactions.
            if 0<index<9:
                config.update(lr=[1e-5,3e-5,1e-4,3e-4][(index-1)//2],
                    adapter_ratio=[1,4][(index-1)%2],batch=16,dropout=.05,
                    warmup_fraction=.03,weight_decay=.01,beta2=.95,schedule='cosine')
            fraction=[0.,.25,.5,.75][index%4]
            strict=index%3==2
            if family=='test_fit':
                indices=fit['fit_screen'];budget=len(indices)*[2,4,6,8][index%4]
            else:
                pub=splits['strict_public_train'] if strict else splits['public_train']
                n_replay=int(256*fraction)
                indices=rng.sample(pub,256-n_replay)+rng.sample(splits['native_replay'],n_replay)
                budget=512
            config.update(replay_fraction=fraction if family=='clean' else None,
                public_quality_view=('strict' if strict else 'quality_v3') if family=='clean' else None)
            jobs.append(dict(folder=str(ROOT/'fast_experiments'/family/f'screen_{index:03d}'),
                config=config,initial_checkpoint=parent,presentations=budget,training_indices=indices,
                validation_suite='fit_screen' if family=='test_fit' else 'screen_validation',
                data_family=family,microbatch=1,max_microbatch=2,checkpoint_presentations=1024,
                gpu_feature_cache=False,intermediate_validation=False))
    paths=[]
    for job in jobs:
        folder=ROOT/'job_specs';folder.mkdir(exist_ok=True)
        family=job['data_family'];name=job['folder'].split('/')[-1]
        path=folder/f'{family}_{name}.json';write(path,job);paths.append(str(path))
    write(ROOT/'queue_plan.json',dict(main_job=str(ROOT/'main_job.json'),fast_jobs=paths,
        screen_experiments=64,parent_baselines=2,
        promotion=dict(clean_top=4,fit_top=4,clean_presentations=4096,fit_full_epochs=4),
        isolation='Never use test-fit checkpoints or test-fit-selected settings for the clean branch.',
        memory_policy='Useful batches and feature cache; reserve headroom, never dummy allocations.'))
    print('QUEUED',len(paths),'fast jobs and one 8-GPU continuation',flush=True)

if __name__=='__main__':main()
