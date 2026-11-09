"""Bound padded decoder tokens, allowing larger batches for shorter sequences."""
def batches(items,token_budget,max_sequences=48):
    group=[];width=0
    for item in items:
        length=item['x']['ids'].shape[1]
        new_width=max(width,length)
        if group and ((len(group)+1)*new_width>token_budget or len(group)>=max_sequences):
            yield group;group=[];width=0
        group.append(item);width=max(width,length)
    if group:yield group
