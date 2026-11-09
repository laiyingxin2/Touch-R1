"""Training parameters only; no changes to Touch-R1 objectives or scoring."""
import math

TARGET_SETS = [
    ['q_proj', 'v_proj'],
    ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
]


def method_spec(config):
    assert config['targets'] in TARGET_SETS
    assert config['rank'] in [4, 8, 16, 32, 64]
    width = config.get('adapter_width', 64)
    assert width in [32, 64, 128, 256]
    return {'name': 'touchreason_parameter_pilot', 'rank': config['rank'],
        'alpha': config.get('lora_alpha', 2*config['rank']),
        'dropout': config['dropout'], 'pool_side': 8,
        'adapter_rank': width, 'targets': config['targets']}


def learning_rate_factor(config, cursor, horizon):
    progress = min(cursor/horizon, 1.)
    warmup = config['warmup_fraction']
    assert 0 <= warmup < 1
    if progress < warmup:
        return progress/warmup
    remaining = (progress-warmup)/(1-warmup)
    schedule = config.get('schedule', 'cosine')
    if schedule == 'constant':
        return 1.
    if schedule == 'linear':
        return 1-remaining
    assert schedule == 'cosine'
    return .5*(1+math.cos(math.pi*remaining))


def signature(config):
    fields = {k: config[k] for k in ['lr', 'rank', 'adapter_ratio', 'batch',
        'dropout', 'weight_decay', 'targets', 'seed', 'warmup_fraction']}
    fields.update(beta2=config.get('beta2', .95), gradient_clip=config.get('gradient_clip', 1.),
        schedule=config.get('schedule', 'cosine'), adapter_width=config.get('adapter_width', 64),
        lora_alpha=config.get('lora_alpha', 2*config['rank']),
        data_mixture=config.get('data_mixture', 'balanced'))
    return fields
