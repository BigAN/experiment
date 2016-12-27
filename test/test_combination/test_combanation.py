import itertools

i = [1, 2, 3, 4]


def cstruct_combination(data):
    '''

    @param data: [(0,uuid),(1,paycc),(2,userid),(3,phone)]
    @return:
    '''
    data = list(enumerate(data))
    def one_l((index, key)):
        return filter(lambda x: x[1] != x[2],
                      [(index, key, x[1]) for x in data])

    return list(itertools.chain(*map(one_l, data)))

rs = cstruct_combination(i)
print rs, len(rs)
