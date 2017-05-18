#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import os
import glob
import argparse
import itertools
import pickle

class cached_property:
    """property whose result is cached"""

    fget = None

    def __init__(self, fget):
        self.fget = fget
        self.__module__ = fget.__module__
        self.__name__ = fget.__name__
        self.__doc__ = fget.__doc__
        self.__cache_key = '__result_cache_{}_{}'.format(
            fget.__name__, id(fget))

    def __get__(self, instance, owner):
        if instance is None:
            return self.fget
        v = getattr(instance, self.__cache_key, None)
        if v is not None:
            return v
        v = self.fget(instance)
        assert v is not None
        setattr(instance, self.__cache_key, v)
        return v


class ByteCnt:
    """counts of byte occurrences at each position"""
    seqlen = None
    """length of the keystream"""

    def __init__(self, cnt):
        self._cnt = cnt

    @classmethod
    def from_file(cls, fpath):
        with open(fpath) as fin:
            if cls.seqlen is None:
                for cnt in itertools.count():
                    if fin.readline().startswith('prob mat'):
                        cls.seqlen = cnt
                        break
                fin.seek(0)

            cnt = np.empty((cls.seqlen, 256), dtype=np.uint64)
            for dst in cnt:
                dst[:] = list(map(int, fin.readline().split()))
        check = cnt.sum(axis=1)
        assert np.all(check == check[0])
        return cls(cnt)

    @property
    def cnt(self):
        """raw count"""
        return self._cnt

    @cached_property
    def nr_sample(self):
        return self._cnt[0].sum()

    @cached_property
    def prob(self):
        """probability mat"""
        return self.cnt.astype(np.longdouble) / self.nr_sample

    @cached_property
    def log_prob(self):
        """log-probability mat"""
        return np.log(self.prob)

    def make_shuffled(self, key):
        """make another testcase by shuffling with given key"""
        if key == 0:
            return self
        ret = np.empty_like(self._cnt)
        for i in range(256):
            ret[:, i] = self._cnt[:, i ^ key]
        return ByteCnt(ret)


class SuccRateCalc:
    _train = None
    """training data"""

    _result = None

    def __init__(self, fdir):
        # meth(testcase) -> cost[seqlen]
        # return the cost of treating given testcase as the keystream
        methods = [
            ('MLE', self._cost_likelihood),
            ('L2', self._cost_l2dist),
            ('KL', self._cost_kl),
        ]

        self._train = ByteCnt.from_file(os.path.join(fdir, 'train.txt'))

        prev_pos = None
        self._result = method_results = {i: [] for (i, _) in methods}

        for fname in sorted(glob.glob(os.path.join(fdir, 'testcase.*'))):
            _, casenum, samples = os.path.basename(fname).split('.')
            samples = int(samples)
            testcase = ByteCnt.from_file(fname)
            assert samples == testcase.cnt[0].sum()
            meth_states = [
                [None, # cost of original keystream
                 np.ones((ByteCnt.seqlen, ), dtype=np.uint32) # succ
                 ]
                for _ in methods]
            for shf in range(256):
                testcase_shuf = testcase.make_shuffled(shf)
                for idx, (name, meth) in enumerate(methods):
                    cost = meth(testcase_shuf)
                    assert cost.shape == (self._train.seqlen, )
                    if not shf:
                        meth_states[idx][0] = cost
                    else:
                        meth_states[idx][1] *= meth_states[idx][0] < cost
            summary = []
            for (_, succ), (name, meth) in zip(meth_states, methods):
                method_results[name].append((samples, succ))
                summary.append(succ[:256].mean())
                if False and summary[-1] <= 0.8:
                    # show bad prob distributions
                    idx = int(np.min(np.nonzero(succ == 0)))
                    print(name, summary[-1], idx)
                    import matplotlib.pyplot as plt
                    plt.plot(np.arange(256), self._train.prob[idx])
                    plt.plot(np.arange(256), testcase.prob[idx])
                    plt.show()
            print('case{} samples={}({:.2f}) succ_rates={}'.format(
                casenum, samples, float(np.log2(samples)), summary))

    def _cost_likelihood(self, testcase):
        # substract by expectation to improve numerical stability
        offset = testcase.nr_sample / 256.0
        return -(((testcase.cnt - offset) * self._train.log_prob).sum(axis=1))

    def _cost_l2dist(self, testcase):
        return (np.square(testcase.prob - self._train.prob)).sum(axis=1)

    def _cost_kl(self, testcase):
        P = self._train
        Q = testcase
        return (P.prob * (P.log_prob - Q.log_prob)).sum(axis=1)

    def write(self, fpath):
        with open(fpath, 'wb') as fout:
            pickle.dump(self._result, fout, pickle.HIGHEST_PROTOCOL)


def main():
    parser = argparse.ArgumentParser(
        description='compute the success rates of recovering a single byte '
        'at various positions/iterations using various methods')
    parser.add_argument('data_directory')
    parser.add_argument('output')
    args = parser.parse_args()
    SuccRateCalc(args.data_directory).write(args.output)

if __name__ == '__main__':
    main()
