"""Padded policy batches and KV rollouts with exact sampled action IDs."""
import torch
import torch.nn.functional as F

def policy_batch(engine, xs, selected=None, logprobs=True):
    lengths=[x['ids'].shape[1] for x in xs];width=max(lengths)
    embeddings=torch.nn.utils.rnn.pad_sequence([engine.embeds(x)[0] for x in xs],batch_first=True)
    positions=torch.cat([F.pad(x['positions'],(0,width-n)) for x,n in zip(xs,lengths)],dim=1)
    mask=torch.arange(width,device=embeddings.device)[None,:]<torch.tensor(lengths,device=embeddings.device)[:,None]
    hidden=engine.base.model(inputs_embeds=embeddings,position_ids=positions,attention_mask=mask,
                             use_cache=False,return_dict=True).last_hidden_state
    values=[]
    if logprobs:
        targets=torch.cat([hidden[i,x['prefix']-1:n-1] for i,(x,n) in enumerate(zip(xs,lengths))])
        labels=torch.cat([x['ids'][0,x['prefix']:] for x in xs])
        parts=[]
        for start in range(0,len(labels),128):
            logits=engine.base.lm_head(targets[start:start+128]).float()
            parts.append(-F.cross_entropy(logits,labels[start:start+128],reduction='none'))
        flat=torch.cat(parts);offset=0
        for x,n in zip(xs,lengths):
            count=n-x['prefix'];values.append(flat[offset:offset+count]);offset+=count
    distributions=[]
    if selected is not None:
        for i,(x,positions_i) in enumerate(zip(xs,selected)):
            z=hidden[i,x['prefix']-1+positions_i]
            distributions.append(F.log_softmax(engine.base.lm_head(z).float(),dim=-1))
    return values,distributions

@torch.no_grad()
def rollout_batch(engine,prompts,group_size,microbatch,max_new_tokens):
    engine.mode(False)
    expanded=[(i,p) for i,p in enumerate(prompts) for _ in range(group_size)]
    result=[[] for _ in prompts]
    eos=engine.tokenizer.eos_token_id
    for start in range(0,len(expanded),microbatch):
        subset=expanded[start:start+microbatch];width=max(p['prefix'] for _,p in subset)
        embeddings=torch.stack([F.pad(engine.embeds(p)[0],(0,0,width-p['prefix'],0)) for _,p in subset])
        positions=torch.cat([F.pad(p['positions'],(width-p['prefix'],0)) for _,p in subset],dim=1)
        lengths=torch.tensor([p['prefix'] for _,p in subset],device='cuda:0')
        mask=torch.arange(width,device='cuda:0')[None,:]>=width-lengths[:,None]
        out=engine.base.model(inputs_embeds=embeddings,position_ids=positions,attention_mask=mask,
                              use_cache=True,return_dict=True)
        tokens_out=[[] for _ in subset];lp_out=[[] for _ in subset];finished=[False]*len(subset)
        limit=min(max_new_tokens,4096-width)
        assert limit>0
        for step in range(limit):
            logits=engine.base.lm_head(out.last_hidden_state[:,-1]).float()
            tokens=torch.multinomial(logits.softmax(-1),1)
            lp=F.log_softmax(logits,-1).gather(-1,tokens).flatten().tolist()
            sampled=tokens.flatten().tolist()
            for j,tok in enumerate(sampled):
                if not finished[j]:
                    tokens_out[j].append(tok);lp_out[j].append(lp[j]);finished[j]=tok==eos
            if all(finished):break
            pos=torch.tensor([p['prefix']+step+p['delta'].item() for _,p in subset],device='cuda:0',dtype=torch.long)
            pos=pos.view(1,-1,1).expand(3,-1,-1)
            mask=torch.cat([mask,torch.ones((len(subset),1),device='cuda:0',dtype=torch.bool)],dim=1)
            out=engine.base.model(input_ids=tokens,position_ids=pos,attention_mask=mask,
                past_key_values=out.past_key_values,use_cache=True,return_dict=True)
        for (i,_),tokens,lps in zip(subset,tokens_out,lp_out):
            result[i].append(dict(tokens=tokens,sampled_logprobs=lps,truncated=tokens[-1]!=eos))
        del out,embeddings,positions,mask
    assert all(len(group)==group_size for group in result)
    return result

def objective_batch(engine,trajectories,config,parity=False):
    from rewarding import segment_mean
    xs=[tr['x'] for tr in trajectories];positions=[tr['positions'] for tr in trajectories]
    lp,intact=policy_batch(engine,xs,positions)
    masked=[]
    for tr in trajectories:
        x=dict(tr['x']);x['features']=x['features'].clone();a,b=x['stream_slices'][tr['stream']]
        x['features'][a:b]=tr['noise'];masked.append(x)
    _,perturbed=policy_batch(engine,masked,positions,logprobs=False)
    losses=[];stats=[]
    for tr,current,p,q in zip(trajectories,lp,intact,perturbed):
        if parity:
            error=(current.detach()-tr['old']).abs()
            assert error.mean().item()<=.02 and torch.allclose(current.detach(),tr['old'],atol=.12,rtol=.01),('old-policy parity',error.max().item(),error.mean().item())
        ratio=(current-tr['old']).exp();advantage=tr['advantage']
        lo,hi=config['clip_bounds'];surrogate=torch.minimum(ratio*advantage,ratio.clamp(lo,hi)*advantage)
        delta=tr['reference']-current;kl=delta.exp()-delta-1
        policy=segment_mean(-surrogate+config['kl_beta']*kl,tr['segments'])
        grounding=(p.exp()*(p-q)).sum(-1).mean()
        loss=policy-config['grounding_gamma']*grounding
        assert torch.isfinite(loss) and torch.isfinite(grounding)
        losses.append(loss)
        stats.append((grounding.detach().item(),segment_mean(kl.detach(),tr['segments']).item(),
                      ((ratio<lo)|(ratio>hi)).float().mean().detach().item()))
    return torch.stack(losses).sum(),stats
