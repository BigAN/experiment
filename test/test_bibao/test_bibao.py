def test(mode, *args):
    d = {"key": 1}

    def test_haha(mode, *args):
        print args
        print d
        print mode
        d['key'] += 1
        d['a']=1
    return test_haha


func = test("online")
func("haha", 1, 43, 4)
func("haha", 1, 43, 4)
func("haha", 1, 43, 4)
