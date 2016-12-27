class User(object):
    a = 1
    def a(self):
        print 'a'

    @staticmethod
    def b():
        print 'b'

    @classmethod
    def c(cls):
        print 'c'

u = User()
u.a(),u.b(),u.c()

User.b()
User.c()

