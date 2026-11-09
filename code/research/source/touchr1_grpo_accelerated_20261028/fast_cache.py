"""Spend remaining VRAM on audited, upcoming tactile features."""
import time
from pathlib import Path
from fast_worker import FeatureCache

class PrefetchCache(FeatureCache):
    def prewarm(self,dataset,orders,cursor,rank,world,max_seconds=2):
        import torch
        started=time.monotonic();position=max(cursor,getattr(self,'prefetch_cursor',cursor));loaded=0
        key=self.engine.feature_key
        roots=[Path('/mnt/workspace/lyx/touchr1_grpo_accelerated_20261028/features')/key,
               Path('/mnt/workspace/lyx/touchreason_v12_best_sft_20261025/features')/key,
               Path('/mnt/workspace/lyx/touchreason_touchr1_three_stage_20261024/features')/key]
        while self.bytes<self.limit-8*2**20 and position<len(orders):
            if time.monotonic()-started>max_seconds:break
            row=dataset.get(orders[position]);position+=world
            for media in row['media']:
                k=media['key']
                if k in self.values:continue
                path=self.engine.cache/(k+'.pt')
                if not path.is_file():path=next((root/(k+'.pt') for root in roots if (root/(k+'.pt')).is_file()),None)
                if path is None:continue
                value=torch.load(path,map_location='cpu',weights_only=True)
                assert value['feature_key']==key and value['media_key']==k
                size=value['features'].numel()*value['features'].element_size()
                if self.bytes+size>self.limit:continue
                try:value=dict(value,features=value['features'].to('cuda:0'))
                except torch.cuda.OutOfMemoryError:
                    self.trim(self.bytes-2**30);torch.cuda.empty_cache();self.prefetch_cursor=position;return loaded
                self.values[k]=value;self.bytes+=size;loaded+=size
        self.prefetch_cursor=position
        return loaded
