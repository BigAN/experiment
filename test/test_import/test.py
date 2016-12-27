class TEST(object):
    def __init__(self):
        self.a = 1
        self.b = 2
        print "???"

ts = [TEST(),TEST(),TEST()]

# print t.a, t.b

def trans(t):
    return [getattr(t,x) for x in [x for x in dir(t) if not x.startswith("__")]]

import multiprocessing as mp
pool = mp.Pool(8)
print pool.map(trans,ts)