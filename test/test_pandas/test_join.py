import pandas as pd
import numpy as np

d = {"a": [1, 2, 3], "c": [4, 5, 6]}
a = pd.DataFrame(data=d)
b = pd.DataFrame(data={"a": [1, 2, 3], "c": [4, 2, 3]})
c = pd.DataFrame(data={"a": [1, 2, 3], "c": [5, 1, 3]})

d = pd.DataFrame(data={"a": [1, 2, 3], "c": [5, 1, 3]})

rs = a.merge(b, on="a", suffixes=("_a", "_b")).merge(c, on='a').merge(d, on='a')
# .merge(c,on="a")
print a
print a.apply(lambda x:x['a']**2,axis=1)

ls = [a, b, c, d]


def mult_merge(l, key):
    x1 = l[0]
    for x in l[1:]:
        x1 = x1.merge(x, on=key)
    return x1

print mult_merge(ls,"a")