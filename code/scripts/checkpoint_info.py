"""Inspect the adapter metadata and safetensors header without loading a GPU."""
import argparse
import json
from pathlib import Path
import struct


def inspect(folder):
    folder = Path(folder)
    manifest = json.loads((folder / 'manifest.json').read_text())
    with (folder / 'adapter_model.safetensors').open('rb') as stream:
        length = struct.unpack('<Q', stream.read(8))[0]
        if length > 32 * 1024**2:
            raise ValueError('Unexpected safetensors header size')
        tensors = json.loads(stream.read(length))
    tensors.pop('__metadata__', None)
    return {'base_model': manifest['base_model'], 'format': manifest['format'],
            'method_spec': manifest['method_spec'], 'language_adapter_tensors': len(tensors),
            'trainable_parameters': manifest['trainable_parameters'],
            'vision_adapter_present': (folder / 'vision_adapter.pt').is_file(),
            'tactile_adapter_present': (folder / 'tactile_adapter.pt').is_file(),
            'historical_loader_state_present': (folder / 'adaptation.pt').is_file(),
            'standalone_base_model_included': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect(args.checkpoint), ensure_ascii=False, indent=2))
