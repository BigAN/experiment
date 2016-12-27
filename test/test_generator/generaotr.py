i = range(1,19,1)


tmp1 = lambda x:x*2
tmp2 = lambda x:x*3

a = (tmp1(x) for x in i)

b = (tmp2(x) for x in a)


while 1:
    print b.next()