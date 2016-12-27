def foo1(tmp=None):
    def foo():
        if tmp:
            print tmp
        else:
            tmp = None

    return foo()


def foo1(tmp=None):
    def foo():
        tmp = 1 + 1
        print tmp

    return foo()


# print foo1(2)

def foo3(tmp=None):
    def foo():
        c = tmp
        tmp = 1 + 1

    return foo()

print foo1(2)
