import numpy as np
import pickle
import argparse
import re
from glob import glob
from os.path import join, split


def get_args():
    '''This function parses and return arguments passed in'''
    # Assign description to the help doc
    parser = argparse.ArgumentParser(
        description='Control execution of Hw2.py')
    # Add arguments
    parser.add_argument(
        '-d', '--debug', type=bool, help='Debug or not',
        default=False, required=False)
    parser.add_argument(
        '-s', '--samples', type=bool, help='Load Samples or not',
        default=False, required=False)
    parser.add_argument(
        '-c', '--clsfr', type=bool, help='Load Classifier or not',
        default=False, required=False)
    args = parser.parse_args()

    return args.debug, args.samples, args.clsfr


def cache_results(score, params):
    R = {}
    R["Classifier"] = params["classifier"]
    R["Overlap_Thresh"] = params["thresh"]
    R["Kmeans"] = params["n_cluster"]
    R["Feature"] = params["feature"]
    R["Score"] = score
    R["Mean"] = np.mean(score)
    Results = pickle.load(open(".Results_Cache", 'r'))
    Results.append(R)
    pickle.dump(Results, open(".Results_Cache", 'w'))
    return Results


def print_(verbose, msg):
    if verbose:
        print(msg)


def get_groundname(name):
    im_num = int(re.findall('\d+', name)[0])
    ground = re.findall('train|test', name)[0]
    groundname = glob(join("face_%sing_groundtruth" % (ground),
                           "*mask%d.png" % im_num))[0]
    return groundname
