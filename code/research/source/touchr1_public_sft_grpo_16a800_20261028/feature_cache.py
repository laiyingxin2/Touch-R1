"""Keep newly computed features local to each pod; reuse verified shared features."""
from pathlib import Path
import os
import shutil
import types

def install(engine):
    ram=Path('/dev/shm/lyx_grpo16_features')/engine.feature_key
    disk=Path('/tmp/lyx_grpo16_features')/engine.feature_key
    ram.mkdir(parents=True,exist_ok=True);disk.mkdir(parents=True,exist_ok=True)
    shared=[Path('/mnt/workspace/lyx/touchreason_v12_best_sft_20261025/features')/engine.feature_key,
            Path('/mnt/workspace/lyx/touchreason_touchr1_three_stage_20261024/features')/engine.feature_key]
    original=type(engine).feature
    def feature(self,media):
        name=media['key']+'.pt'
        self.cache=ram if (ram/name).exists() or shutil.disk_usage(ram).free>=24*2**30 else disk
        dest=self.cache/name
        if not dest.exists():
            for directory in shared:
                source=directory/name
                if source.exists():
                    temp=dest.with_suffix(f'.{os.getpid()}.tmp')
                    shutil.copyfile(source,temp);temp.replace(dest);break
        return original(self,media)
    engine.cache=ram
    engine.feature=types.MethodType(feature,engine)
