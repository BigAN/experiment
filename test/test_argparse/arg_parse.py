import argparse

TEST = "test"
ONLINE = "online"


def init_arguments():
    def right_mode(p):
        if p == TEST or p == ONLINE:
            return p
        else:
            raise argparse.ArgumentError('%s is not a mode.' % p)

    parser = argparse.ArgumentParser()

    parser.add_argument('-d', type=str, dest="date",
                      help="target date")
    parser.add_argument('-m', type=str, dest='mode',
                      help="work mode")
    parser.add_argument('-k', type=str, dest="key",
                      help="map key(rider or rider_poi)")
    return parser.parse_args()


if __name__ == "__main__":
    args = init_arguments()
    print args.mode
