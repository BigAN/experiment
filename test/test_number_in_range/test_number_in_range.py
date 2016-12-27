def trans_value_to_threds(inp_list, tgt, lo=0, hi=None):
    if lo < 0:
        raise ValueError("lo must be non-negative")
    if hi is None:
        hi = len(inp_list)
    if tgt > inp_list[hi - 1] or tgt < inp_list[lo]:
        return None
    if tgt == inp_list[lo]:
        return inp_list[0], inp_list[1]
    if tgt == inp_list[hi - 1]:
        return inp_list[hi - 2], inp_list[hi - 1]

    while lo < hi:
        mid = (lo + hi) // 2
        if inp_list[mid] < tgt:
            lo = mid + 1
        else:
            hi = mid

    return inp_list[lo - 1], inp_list[lo]

inp = [x/100.0 for x in range(0,101,5)]
print inp
print trans_value_to_threds(inp,1)


l = [0.1,0.2,0.3,0.5]
print map(lambda x:("_".join(map(str,trans_value_to_threds(inp,x))),1),l)