l = ["a", 'b', 'c']


class MyClass():
    def __init__(self):
        _vars = ["a", "b", "c"]
        for _var in _vars:
            self.__dict__[_var] = dict()

    def __str__(self):
        print self.a



s = MyClass()


# print s.a

class testOder(object):
    a = 1
    b = 2

    def __init__(self, name):
        self.a = 1
        self.b = 2
        print name
        print "aaaaa"
        print self.a
        print self.__dict__
        print self.__getattribute__()
        print self.__
        print "aaaa"
        # return 0
        print self.__dict__

a = testOder(1)
# print a
