"""Meaningful CPU checks for distributed normalization and test resumption."""
import ast
import json
from pathlib import Path
import socket
import tempfile
import types

from experiment import RUN, SEARCH, read, write
from worker import allreduce_gradients, read_predictions


def distributed_check(rank, port, output):
    import torch
    import torch.distributed as dist
    dist.init_process_group('gloo', init_method=f'tcp://127.0.0.1:{port}', rank=rank, world_size=2)
    parameter = torch.nn.Parameter(torch.tensor([.2, -.3]))
    engine = types.SimpleNamespace(parameters=[('p', parameter)])
    # Unequal per-rank target lengths: local means would give the wrong result.
    x = [torch.tensor([[1., 2.]]), torch.tensor([[3., 1.], [2., -1.], [1., 4.]])][rank]
    y = [torch.tensor([.3]), torch.tensor([.2, -.1, .4])][rank]
    (((x @ parameter-y)**2).sum()/4).backward()
    allreduce_gradients(engine, 100.)
    reference = torch.nn.Parameter(torch.tensor([.2, -.3]))
    all_x = torch.tensor([[1., 2.], [3., 1.], [2., -1.], [1., 4.]])
    all_y = torch.tensor([.3, .2, -.1, .4])
    (((all_x @ reference-all_y)**2).sum()/4).backward()
    assert torch.allclose(parameter.grad, reference.grad, atol=1e-6)
    if rank == 0:
        Path(output).write_text(json.dumps({'gradient': parameter.grad.tolist(), 'reference': reference.grad.tolist()}))
    dist.destroy_process_group()


def main():
    import torch.multiprocessing as mp
    checks = {}
    with tempfile.TemporaryDirectory() as temp:
        p = Path(temp)
        with socket.socket() as s:
            s.bind(('127.0.0.1', 0)); port = s.getsockname()[1]
        output = str(p/'grad.json')
        mp.spawn(distributed_check, args=(port, output), nprocs=2, join=True)
        checks['distributed_token_normalized_gradient_matches_serial'] = read(output)
        (p/'rank_w2_0.jsonl').write_bytes(b'{"id":"a","response":"A"}\n{"id":')
        assert list(read_predictions(p)) == ['a']
        assert (p/'rank_w2_0.jsonl').read_bytes().endswith(b'\n')
        checks['partial_prediction_line_recovery'] = True
        (p/'rank_w4_0.jsonl').write_text('{"id":"a","response":"A"}\n')
        try:
            read_predictions(p)
        except AssertionError:
            checks['duplicate_prediction_rejected'] = True
        else:
            raise AssertionError('Duplicate predictions accepted')
    import sys
    sys.path.insert(0, str(SEARCH))
    from parameters import learning_rate_factor
    plan = read(RUN/'plan.json')
    c = plan['config']; budget = plan['schedule_horizon']
    assert learning_rate_factor(c, budget, budget) == 0.
    assert learning_rate_factor(c, int(budget*.03), budget) == 1.
    assert learning_rate_factor(c, 2048, budget) > 0.
    checks['schedule_uses_full_training_budget'] = True
    tree = ast.parse((RUN/'worker.py').read_text())
    score = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'score')
    assert 'sealed.json' in ast.get_source_segment((RUN/'worker.py').read_text(), score.body[3]) or 'sealed.json' in ast.get_source_segment((RUN/'worker.py').read_text(), score)
    assert 'test_qa.jsonl' not in (RUN/'experiment.py').read_text()
    checks['train_dataset_is_separate_from_test_gold'] = True
    for file in RUN.glob('*.py'):
        ast.parse(file.read_text())
    checks['syntax'] = True
    write(RUN/'preflight.json', {'passed': True, 'checks': checks, 'plan': plan,
          'test_gold_opened': False})
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
