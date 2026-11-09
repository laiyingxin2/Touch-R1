"""Independent continuation and validation experiments from one immutable parent."""
import collections
import datetime
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parent
V12 = Path('/mnt/workspace/lyx/touchreason_v12_best_sft_20261025')
PARENT = Path('/mnt/workspace/lyx/touchreason_touchr1_three_stage_20261024')
PUBLIC = Path('/mnt/workspace/lyx/public_tactile_qa_20261022')
PY = sys.executable
sys.path[:0] = [str(V12), str(PARENT), str(PUBLIC), str(PARENT / 'small_parameter_search_20261024')]
from experiment import read, write, digest, identity


def emit(folder, event, **values):
    value = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), event=event, **values)
    print(json.dumps(value, ensure_ascii=False), flush=True)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'events.jsonl').open('a') as f:
        f.write(json.dumps(value, ensure_ascii=False) + '\n')


class Dataset:
    def __init__(self, family='clean'):
        assert family in ['clean', 'test_fit']
        meta = read(ROOT / ('data_manifest.json' if family == 'clean' else 'test_fit_dataset_manifest.json'))
        path = Path(meta['runtime_database'])
        assert path.stat().st_size == meta['database_bytes']
        self.con = sqlite3.connect(f'file:{path}?mode=ro&immutable=1', uri=True)

    def get(self, idx):
        value = self.con.execute('SELECT record FROM examples WHERE idx=?', (idx,)).fetchone()
        assert value is not None
        return json.loads(value[0])


def make_engine(config, checkpoint):
    import torch
    from experiment import engine_for
    plan = read(V12 / 'plan.json')
    engine = engine_for(plan)
    aliases_path=ROOT/'verified_grpo_source_aliases.json'
    if aliases_path.exists():
        aliases=read(aliases_path)
        for old,alias in aliases.items():
            ep=engine.episodes[alias['episode_id']]
            assert ep['source']['path']==old
            assert alias['verification']=='all original 64x64 BGR INTER_AREA SHA256 frame fingerprints'
            ep['source']=dict(ep['source'],path=alias['path'])
            ep['verified_alias_original_path']=old
            if ep['source']['kind']=='png_sequence':
                assert set(alias['frame_sha256'])==set(ep['source']['filenames'])
                ep['verified_alias_frame_sha256']=alias['frame_sha256']
            else:
                assert ep['source']['kind']=='video' and ep['source']['frame_count']==alias['verified_frame_count']
                ep['verified_alias_video_sha256']=alias['video_sha256']
    engine.load(checkpoint)
    if config.get('decoder_dtype') == 'float32':
        # Keep the frozen tactile encoder and its cached features unchanged.
        # FP32 decoder arithmetic avoids shape-dependent BF16 rollout drift.
        engine.base.model.float()
        engine.base.lm_head.float()
        runtime = read(ROOT / 'runtime.json') if (ROOT / 'runtime.json').exists() else {}
        torch.backends.cuda.matmul.allow_tf32 = (config.get('allow_tf32', False)
                                                and not runtime.get('disable_tf32', False))
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        emit(ROOT, 'decoder_precision', dtype='float32',
             allow_tf32=torch.backends.cuda.matmul.allow_tf32,
             allocated_gib=torch.cuda.memory_allocated()/2**30)
    # Rank, targets, Stage-1 encoder and the SFT objective stay fixed. Dropout
    # is a runtime training hyperparameter; saved tensor names/shapes match.
    for module in engine.peft.modules():
        drop = getattr(module, 'lora_dropout', None)
        if drop is not None:
            for layer in drop.values():
                if hasattr(layer, 'p'):
                    layer.p = config.get('dropout', .05)
    engine.peft.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    assert sum(p.numel() for _, p in engine.parameters) == 81209920
    from feature_cache import install
    install(engine)
    return engine


def save_model(engine, path, config, state=None):
    engine.save(path, state)
    write(Path(path) / 'continuation_config.json', config)
    write(Path(path) / 'parent_identity.json', read(ROOT / 'parent_checkpoint_frozen.json'))


def table_metrics(native):
    """Literal local task mapping; do not invent an official Avg or CSC."""
    from native_scoring import summarize
    report = summarize(native)
    attrs, tasks = report['attributes'], report['tasks']
    value = {}
    for name, attribute in [('H-Acc', 'hardness'), ('R-Acc', 'roughness'), ('P-Acc', 'protrusion'), ('Mat-Acc', 'material')]:
        value[name] = 100 * attrs[attribute]['accuracy'] if attribute in attrs else None
    for name, task, scale in [('OMAE', 'ordinal', 1), ('L2-EM', 'joint_em', 100), ('SFD', 'pair', 100), ('SOI-rho', 'ranking', 1)]:
        value[name] = scale * tasks[task]['score'] if task in tasks else None
    value.update(CSC=None, Avg=None)
    normalized = {k: v for k, v in value.items() if v is not None}
    if value['OMAE'] is not None:
        normalized['OMAE'] = 100 * (1 - value['OMAE'])
    if value['SOI-rho'] is not None:
        normalized['SOI-rho'] = 100 * value['SOI-rho']
    # User-requested arithmetic mean over measured, direction-normalized columns.
    # It is not asserted to reproduce the published full-metric-set Avg=60.1.
    value['Avg'] = sum(normalized.values()) / len(normalized) if normalized else None
    return dict(values=value, denominators=dict(attributes={a: v['n'] for a, v in attrs.items()},
                tasks={t: v['n'] for t, v in tasks.items()}),
                protocol='Local reconstructed task mapping, not official TouchReason-Bench equivalence.',
                missing={'CSC': 'No independently verified same-physical-object cross-sensor pairing.'},
                average=dict(definition='User-requested arithmetic mean of available normalized columns.',
                  normalized_columns=normalized, n_columns=len(normalized),
                  OMAE_transform='100*(1-OMAE)', SOI_transform='100*rho',
                  missing_CSC_excluded=True, official_paper_Avg_equivalence=False))


def evaluate_validation(engine, dataset, indices, folder, controls=('real',), evaluation_split='val'):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    selected = [dataset.get(i) for i in indices]
    assert evaluation_split in ['val', 'test_fit']
    assert all(r['split'] == evaluation_split for r in selected)
    write(folder / 'evaluation_manifest.json', dict(indices=indices, split=evaluation_split, n=len(selected),
          controls=list(controls), max_new_tokens=384, decoding='greedy', test_used=evaluation_split == 'test_fit',
          same_data_train_and_eval=evaluation_split == 'test_fit'))
    engine.mode(False)
    # Donors come from validation inputs only and exclude all in-question media.
    donors = {m['key']: m for r in selected for m in r['media']}
    for control in controls:
        path = folder / (control + '.jsonl')
        done = [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
        assert [p['id'] for p in done] == [r['id'] for r in selected[:len(done)]]
        with path.open('a') as stream:
            for i, row in enumerate(selected[len(done):], len(done) + 1):
                if control == 'mismatch':
                    forbidden = {m['key'] for m in row['media']}
                    candidates = [m for k, m in sorted(donors.items()) if k not in forbidden]
                    assert candidates
                    for j, media in enumerate(row['media']):
                        engine.mismatch[media['key']] = candidates[j % len(candidates)]
                response, tokens = engine.generate(row, control=control, max_new_tokens=384)
                stream.write(json.dumps(dict(id=row['id'], response=response, control=control,
                                             token_ids=tokens), ensure_ascii=False) + '\n')
                stream.flush()
                if i == 1 or i % 16 == 0 or i == len(selected):
                    emit(folder, 'validation_prediction', control=control, n=i, total=len(selected))
        write(folder / (control + '_sealed.json'), dict(n=len(selected), sha256=digest(path)))
    return score_validation(dataset, indices, folder, controls, evaluation_split)


def score_validation(dataset, indices, folder, controls=('real',), evaluation_split='val'):
    from scoring import score_predictions
    from native_scoring import parse_result
    from rewarding import answer
    folder = Path(folder)
    selected = [dataset.get(i) for i in indices]
    assert all(r['split'] == evaluation_split for r in selected)
    native_ids = {r['native_id'] for r in selected if r['dataset'] == 'touchreason_bench'}
    gold_file = ROOT / ('data/native_validation_gold.jsonl' if evaluation_split == 'val' else 'test_fit_data/gold.jsonl')
    gold = [json.loads(s) for s in gold_file.read_text().splitlines()]
    gold = [r for r in gold if r['qa_id'] in native_ids]
    public_gold = [r['public_row'] for r in selected if r['task'] == 'public_json']
    result = {}
    for control in controls:
        path = folder / (control + '.jsonl')
        assert read(folder / (control + '_sealed.json'))['sha256'] == digest(path)
        predictions = [json.loads(s) for s in path.read_text().splitlines()]
        metric = score_predictions(predictions, gold, public_gold)
        by_id = {r['id']: r['response'] for r in predictions}
        native = [parse_result(r, answer(by_id['touchreason_bench/' + r['qa_id']])) for r in gold]
        tasks = collections.defaultdict(list)
        for r in native:
            exact = r['correct'] if r['task'] != 'ranking' else '|'.join(s.strip() for s in r['output'].split('|')) == r['answer']
            tasks[r['task']].append(bool(exact))
        task_accuracy = {k: sum(v)/len(v) for k, v in tasks.items()}
        metric['native_task_macro_exact_accuracy'] = sum(task_accuracy.values())/len(task_accuracy) if tasks else None
        metric['native_task_exact_accuracy'] = task_accuracy
        metric['table2_local'] = table_metrics(native) if native else None
        public_values = [v for k, v in metric['exact_accuracy_per_dataset'].items() if k != 'touchreason_bench']
        components = ([metric['native_task_macro_exact_accuracy']] if tasks else []) + public_values
        metric['local_development_score'] = sum(components)/len(components)
        metric['selection_metric_is_not_paper_Avg'] = True
        metric['test_used_for_training_or_selection'] = evaluation_split == 'test_fit'
        metric['same_data_train_and_eval'] = evaluation_split == 'test_fit'
        metric['generalization_benchmark_result'] = False if evaluation_split == 'test_fit' else None
        result[control] = metric
    write(folder / 'metrics.json', result)
    return result
