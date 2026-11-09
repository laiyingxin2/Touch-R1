"""Keep full-data training disabled while small-data SFT is explored."""
import json
from pathlib import Path


POLICY_PATH = Path(__file__).resolve().parent / 'training_policy.json'


def require_full_training_allowed(stage, policy_path=None):
    path = Path(policy_path) if policy_path is not None else POLICY_PATH
    policy = json.loads(path.read_text()) if path.exists() else {}
    if stage == 'grpo' and policy.get('grpo_enabled') is not True:
        raise RuntimeError('GRPO is disabled: continue small-data SFT exploration.')
    if policy.get('full_data_training_allowed') is not True:
        raise RuntimeError(
            'Full-data training is held: continue small-data SFT exploration '
            'until validation quality is adequate. See ' + str(path))
