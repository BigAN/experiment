class hah(object):
    def __init__(self):
        pass

    def _haha(self):
        print "hha"

    def _ouou(self):
        print "ouou"

    def cal(self):
        print self.__dict__.items()


h = hah()
print hah.__dict__
print h.cal()