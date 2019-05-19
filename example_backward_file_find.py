#!/usr/bin/env python

import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description='recursively search for a file from a given file location to root')
    parser.add_argument('-d', '--directory', type=str, default=os.getcwd(),
                        help='path to begin search from. default cwd')

    args = parser.parse_args()

    return args


def main():

    args = parse_args()

    path = os.path.realpath(args.directory)
    while path != "/":
        print(path)
        if os.path.isdir(os.path.realpath(os.path.join(path, '.idea'))):
            print(os.path.basename(path))
            return None
        else:
            path = os.path.realpath(os.path.join(path, '..'))
    print('Nothing found')


if __name__ == "__main__":
    main()

