def list_to_bigrams(somelist):
    it = iter(somelist)
    old = next(it, None)
    for new in it:
        yield old, new
        old = new


print list(list_to_bigrams([1,2,3,4]))




def output(somelist):
    it = iter(somelist)
    old = it.next()
    for new in it:
        yield old,new
        old = new