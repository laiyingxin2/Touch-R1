"""Method ablations sharing the original frozen backbone, input and decoder."""
from runtime import *
import re
from score_quality_v3 import scales

class MethodEngine(Engine):
    def __init__(self,spec):
        self.spec=spec
        setup()
        if MODEL_LOAD_PATH!=MODEL:
            identity=json.loads((Path(MODEL_LOAD_PATH)/'source_identity.json').read_text())
            assert identity['source_model']==MODEL and identity['verified_local_sha256']
            assert all((Path(MODEL_LOAD_PATH)/n).stat().st_size==v['bytes'] for n,v in identity['files'].items())
        self.processor=AutoProcessor.from_pretrained(MODEL_LOAD_PATH,local_files_only=True,min_pixels=IMAGE_SIDE**2,max_pixels=IMAGE_SIDE**2)
        self.tokenizer=self.processor.tokenizer
        self.base=Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_LOAD_PATH,torch_dtype=torch.bfloat16,
                  device_map={'':'cuda:0'},attn_implementation='sdpa',local_files_only=True)
        self.base.requires_grad_(False)
        kwargs={}
        if spec.get('last_layers'):
            kwargs.update(layers_to_transform=list(range(self.base.config.num_hidden_layers-spec['last_layers'],self.base.config.num_hidden_layers)),layers_pattern='layers')
        rank=spec.get('rank',8)
        self.peft=get_peft_model(self.base,LoraConfig(r=rank,lora_alpha=spec.get('alpha',2*rank),lora_dropout=spec.get('dropout',.05),
                  target_modules=spec.get('targets',['q_proj','v_proj']),bias='none',task_type='CAUSAL_LM',
                  use_dora=spec.get('dora',False),**kwargs))
        self.peft.peft_config['default'].base_model_name_or_path=MODEL
        if spec.get('fixed_a'):
            for n,p in self.peft.named_parameters():
                if 'lora_A.' in n:p.requires_grad_(False)
        self.adapter=(ResidualAdapter(rank=spec.get('adapter_rank',64)).to('cuda:0',dtype=torch.bfloat16)
                      if spec.get('adapter_rank',64) else torch.nn.Identity())
        self.base.config.use_cache=False
        self.peft.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        # Non-reentrant checkpointing supports frozen inputs. Avoid forcing input
        # gradients through frozen lower layers in the last-layers-only ablation.
        self.media={r['id']:r for r in read_jsonl(DATA/'organized/media_index.jsonl')}
        self.memory={};CACHE.mkdir(exist_ok=True,parents=True)
        language=[(n,p) for n,p in self.peft.named_parameters() if p.requires_grad]
        assert language and all('lora_' in n and 'visual' not in n for n,_ in language)
        self.parameters=language+[('tactile_adapter.'+n,p) for n,p in self.adapter.named_parameters()]
        self.all_adaptation=[(n,p) for n,p in self.peft.named_parameters() if 'lora_' in n]
        self.all_adaptation += [('tactile_adapter.'+n,p) for n,p in self.adapter.named_parameters()]
        self.fa_corrections={}
        if spec.get('fixed_a'):
            named=dict(self.peft.named_parameters())
            for n,p in language:
                if 'lora_B.' in n:
                    a=named[n.replace('lora_B.','lora_A.')].detach().float()
                    gram=a@a.T
                    # Paper Eq.6 transposed to PEFT's B(out,r) @ A(r,in)
                    # convention; applied to the gradient before AdamW.
                    inv=torch.linalg.solve(gram,torch.eye(rank,device=gram.device))*(rank/(2*rank))**2
                    self.fa_corrections[n]=inv
        emit('method_model_loaded',method=spec['name'],spec=spec,trainable=sum(p.numel() for _,p in self.parameters),
             language_trainable=sum(p.numel() for _,p in language),adapter_trainable=sum(p.numel() for p in self.adapter.parameters()),
             stored_adaptation_parameters=sum(p.numel() for _,p in self.all_adaptation),
             total=sum(p.numel() for p in self.peft.parameters())+sum(p.numel() for p in self.adapter.parameters()),
             feature_key=FEATURE_KEY,allocated_gib=torch.cuda.memory_allocated()/2**30)

    def snapshot(self):return {n:p.detach().cpu().clone() for n,p in self.all_adaptation}
    @torch.no_grad()
    def restore(self,state):
        assert set(state)=={n for n,_ in self.all_adaptation}
        for n,p in self.all_adaptation:
            assert state[n].shape==p.shape
            p.copy_(state[n].to(device=p.device,dtype=p.dtype))
    def save(self,path,state=None):
        path=Path(path);path.mkdir(exist_ok=True,parents=True)
        self.peft.save_pretrained(path)
        tmp=path/'adaptation.tmp';torch.save(self.snapshot(),tmp);tmp.replace(path/'adaptation.pt')
        torch.save(self.adapter.state_dict(),path/'tactile_adapter.pt')
        if state is not None:
            tmp=path/'trainer_state.tmp';torch.save(state,tmp);tmp.replace(path/'trainer_state.pt')
        write_json(path/'manifest.json',{'base_model':MODEL,'feature_key':FEATURE_KEY,'feature_spec':FEATURE_SPEC,
            'method_spec':self.spec,'format':'method-search-v1','seed':SEED,'complete':True,
            'trainable_parameters':sum(p.numel() for _,p in self.parameters)})
    def load(self,path):
        path=Path(path);manifest=json.loads((path/'manifest.json').read_text())
        assert manifest['feature_key']==FEATURE_KEY and manifest['method_spec']==self.spec
        self.restore(torch.load(path/'adaptation.pt',map_location='cpu',weights_only=True))
    @torch.no_grad()
    def correct_gradients(self):
        for n,p in self.parameters:
            if n in self.fa_corrections and p.grad is not None:
                p.grad.copy_((p.grad.float()@self.fa_corrections[n]).to(p.grad.dtype))
    def groups(self,language_lr,adapter_lr):
        groups=[]
        for n,p in self.parameters:
            lr=adapter_lr if n.startswith('tactile_adapter.') else language_lr
            if self.spec.get('loraplus') and 'lora_A.' in n:lr/=16
            groups.append({'params':[p],'lr':lr,'initial_lr':lr})
        return groups
    def loss(self,row):
        x=self.encode(row);lp=self.logprobs(x)
        if not self.spec.get('attribute_weight'):return -lp.mean()
        answer=row['messages'][-1]['content']
        tokenized=self.tokenizer(answer,add_special_tokens=False,return_offsets_mapping=True)
        assert x['ids'][0,x['prefix']:-1].tolist()==tokenized['input_ids'],'Target offset alignment failed'
        props='|'.join(re.escape(p) for p in scales(row))
        spans=[m.span(1) for m in re.finditer(r'"(?:'+props+r')"\s*:\s*"([^"]*)"',answer)]
        assert spans,'No attribute supervision spans found'
        weights=[float(self.spec['attribute_weight']) if any(a<end and b>start for a,b in spans) else 1.
                 for start,end in tokenized['offset_mapping']]+[1.]
        w=torch.tensor(weights,device=lp.device,dtype=lp.dtype)
        assert w.numel()==lp.numel()
        return -(lp*w).sum()/w.sum()
