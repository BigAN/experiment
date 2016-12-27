config = [['a', '>', '1'], ['b', '<', '2']]

a = [0, 1, 2, 3]
b = [1, 2, 3, 4]


# print a,b
def generate_func(conf):
    return lambda x: eval('x {operater} {value} '.format(
            **{"operater": conf[1], "value": conf[2]}))


def apply_config(config):
    def one_config(conf):
        return filter(generate_func(conf), globals().get(conf[0]))

    return map(one_config, config)


print apply_config(config)
