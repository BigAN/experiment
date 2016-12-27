i = [(1,2),(2,3)]

print map(lambda (x,y):x+y,i)

def double_char(s):
    return "".join(map(lambda x:x*2,s))

print double_char("hello wolrd")