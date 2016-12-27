import itertools
somelists = [[1,2,3],[2,3,4],[7,8,9]]
for element in itertools.product(*somelists):
    print element

print list(itertools.chain(*somelists))
