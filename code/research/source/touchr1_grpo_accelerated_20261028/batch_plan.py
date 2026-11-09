"""Balance prompt context cost while preserving every selected training row."""
import json,math

def partition(dataset,indices,rank,world):
    capacities=[len(indices[r::world]) for r in range(world)]
    bins=[[] for _ in range(world)];loads=[0]*world;weighted=[]
    for position,index in enumerate(indices):
        row=dataset.get(index);frames=0
        for media in row['media']:
            if media['kind']=='images':frames+=len(media['paths'])
            elif media['kind']=='raw':frames+=len(media['frame_indices'])
            elif media['kind']=='public':frames+=min(8,len(media['source'].get('frames',range(8))))
            else:frames+=1
        pool=8 if frames<=32 else 4
        cost=frames*pool*pool+math.ceil(len(json.dumps(row['input'],ensure_ascii=False))/3)+128
        weighted.append((cost,position,index,row))
    for cost,position,index,row in sorted(weighted,key=lambda v:(-v[0],v[1])):
        target=min((r for r in range(world) if len(bins[r])<capacities[r]),key=lambda r:(loads[r],len(bins[r]),r))
        bins[target].append((position,index,row));loads[target]+=cost
    assigned=[index for group in bins for _,index,_ in group]
    assert sorted(assigned)==sorted(indices) and len(assigned)==len(set(assigned))
    return [row for _,_,row in sorted(bins[rank])],dict(estimated_cost=loads,
        indices_by_rank=[[index for _,index,_ in sorted(group)] for group in bins])
