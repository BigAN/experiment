def binary_search(l, x, low=0, high=None):
    '''
    find x in l
    :param l: input number list
    :param x: float
    :return: return low,high
    '''
    high = len(l)
    if low < 0 :
        raise ValueError("lo must be non-negative")
    while low < high:
        mid = (low + high)/2
        print mid,'mid',x,low,high
        if x > l[mid]:
            low = mid + 1
        else:
            high = mid
    return l[low]

print binary_search([0,1,2,3],1)

