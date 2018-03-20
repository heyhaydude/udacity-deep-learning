import optparse
import argparse
"""
Helper script to get command line arguements
"""

""" All keys must be of the same type"""
def sorted_dict2str(dictionary):
    s = "Score, Games\n"
    sc = 0
    for k in sorted(dictionary.keys()):
        while sc < k:
            s += "{:5}, {:5}\n".format(sc, 0)
            sc += 1
        v = dictionary[k]
        s += "{:5}, {:5}\n".format(k, v)
        sc += 1
    return s


def str2bool(v):
    if v.lower() in ['yes', 'true', 't', 'y', '1']:
        return True
    elif v.lower() in ['no', 'false', 'f', 'n', '0']:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_arguments():
    parser = optparse.OptionParser()

    parser.add_option('-t', '--training',
        action="store", dest="training",
        help="Training flag.", default="True")

    parser.add_option('-e', '--env',
        action="store", dest="env",
        help="Training Environment.", default="snake")

    parser.add_option('-v', '--verbose',
        action="store_true", dest="v",
        help="Verbose mode.", default=False)

    parser.add_option('--fps', '--fr',
        action="store", dest="fps",
        help="Frame rate when not training.", default=30)

    parser.add_option('--width',
        action="store", dest="W",
        help="Input frame width.", default=100)

    parser.add_option('--height',
        action="store", dest="H",
        help="Input frame height.", default=100)

    options, args = parser.parse_args()
    return (options.training, options.env, options.v,
            int(options.fps), int(options.W), int(options.H))
