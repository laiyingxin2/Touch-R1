"""Explicit loader contract for the quick candidate release.

Candidates retain useful weak source hypotheses without silently activating
them as verified labels. The caller resolves `evidence_locators` to images;
only `model_text` is serialized as text in the VLM message.
"""
import json
from pathlib import Path


def prepare(record, allow_unverified_source=False, release_directory=None,
            target_mode='answer'):
    if record.get('label_tier')=='causal_issue_pending_successful_rewrite':
        raise ValueError('unsupported_causal_rationale_requires_correction')
    if not record.get('strict_eligible') and not allow_unverified_source:
        raise ValueError('candidate_not_qualified_for_default_training')
    q=record['model_input']
    if target_mode not in ('answer', 'answer_with_evidence'):
        raise ValueError('unknown_target_mode')
    text={'question':q['question'],'options':q['options'],
          'inputs':[f'Input {i+1}' for i in range(len(record['evidence_locators']))]}
    # Only public, audited representation metadata enters the prompt. Private
    # source paths, episode names and reviewer answers remain outside it.
    for key in ('frames', 'representation', 'input_definitions', 'recorded_frame_indices'):
        if key in q:text[key]=q[key]
    locators=[]
    for source in record['evidence_locators']:
        loc=dict(source)
        if release_directory is not None and loc.get('image_paths'):
            loc['image_paths']=[str(Path(release_directory)/p) for p in loc['image_paths']]
        locators.append(loc)
    target=record['candidate_answer']
    rationale=record.get('rationale_target')
    if target_mode=='answer_with_evidence':
        if not rationale or not record.get('machine_consensus_candidate'):
            raise ValueError('reviewed_evidence_target_unavailable')
        if rationale['conclude']!=(q.get('options') or {}).get(target):
            raise ValueError('evidence_target_answer_mismatch')
        target=json.dumps({'evidence':rationale['perceive'],
                           'comparison':rationale['compare'],
                           'answer':target,'conclusion':rationale['conclude']},ensure_ascii=False)
        text['response_format']={'evidence':'frame-cited visible facts',
                                 'comparison':'brief comparison, empty for a single-frame observation',
                                 'answer':'one option letter','conclusion':'selected option text'}
    return {'model_text':json.dumps(text,ensure_ascii=False),
            'evidence_locators':locators, 'target':target,
            'target_mode':target_mode,
            'physical_reward_enabled':False,
            'experimental_weak_source_opt_in':not record.get('strict_eligible',False),
            'label_tier':record['label_tier']}


def reward_answer(record):
    if not record.get('strict_eligible') or not record.get('physical_reference_verified'):
        raise ValueError('unverified_physical_reward_is_disabled')
    return record['candidate_answer']
