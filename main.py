#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from cryptopals.utils import discover_challenges

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('challenges', nargs='*',
                        help='names of challenges to run; leave empty for all'
                        ' challenges')
    args = parser.parse_args()

    all_ch = discover_challenges()
    ch = args.challenges
    if not ch:
        ch = all_ch
    else:
        all_ch = {i.__name__: i for i in all_ch}
        ch = [all_ch[i] for i in ch]

    for i in ch:
        print('Run {}'.format(i.__name__), end='', flush=True)
        ret = i()
        if ret:
            print(': ', end='')
            print(repr(ret), end='')
        print()

if __name__ == '__main__':
    main()
