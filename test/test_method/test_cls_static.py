class User(object):
    a = 1

    def __init__(self):
        c = 2

    # @staticmethod
    # def b():
    #     print self.a

    @classmethod
    def c(cls):
        print cls.c


u = User()

# print User.a

# u.b()

User.c()
print u

print "haha".format()