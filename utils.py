# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division

import sys

from os import makedirs
from os.path import abspath, exists, join, split

from scipy.io import loadmat, savemat

import numpy as np

import shelve

import cv2

DETECTOR = cv2.xfeatures2d.SURF_create()
DESCRIPTOR = DETECTOR

def load_index(filename):
    fh = shelve.open(filename, 'r')
    if not 'index' in fh:
        raise IOError('not an index file')
    index = fh['index']
    fh.close()
    return index


def save_index(index, filename):
    fh = shelve.open(filename, flag='n', protocol=2)
    fh['index'] = index
    fh.close()


def load_data(filename):
    idict = loadmat(filename, appendmat=False, squeeze_me=True)
    if idict['is_custom_dict'] == 1:
        idict.pop('is_custom_dict')
        idict.pop('__globals__')
        idict.pop('__header__')
        idict.pop('__version__')
        return idict
    return idict['data']


def save_data(data, filename, force_overwrite=False):
    # if dir/subdir doesn't exist, create it
    dirname = split(filename)[0]
    if not exists(dirname):
        makedirs(dirname)

    if isinstance(data, dict):
        data.update({'is_custom_dict': 1})
        savemat(filename, data, appendmat=False)
        data.pop('is_custom_dict')
    else:
        savemat(filename, {'is_custom_dict': 0, 'data': data}, appendmat=False)


def kp2arr(kps):
    '''cv2.KeyPoint to np.array so that we can use pickle to save the model to disk'''
    return np.array([(kp.pt[0], kp.pt[1], kp.size, kp.angle) for kp in kps])


def arr2kp(arr):
    '''np.array to cv2.KeyPoint'''
    return [cv2.KeyPoint(x, y, size, angle) for (x, y, size, angle) in arr]


def get_random_sample(imlist, impath, n_samples=int(1e5), n_per_file=100, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState()

    n_images = len(imlist)

    n = 0
    samples = []
    while n < n_samples:
        i = random_state.randint(0, n_images)
        imfile = abspath(join(impath, imlist[i]))
        if not exists(imfile):
            print('{} do not exists'.format(imfile))
            continue

        im = cv2.imread(imfile, cv2.IMREAD_GRAYSCALE)
        kp = DETECTOR.detect(im)
        kp, desc = DESCRIPTOR.compute(im, kp)

        n_desc = len(desc)
        n_per_file_ = min(n_per_file, n_desc)
        idxs = random_state.choice(n_desc, n_per_file_, replace=False)
        samples.append(desc[idxs, :])
        n += len(idxs)

        print('\r {}/{} samples'.format(n, n_samples), end='')
        sys.stdout.flush()

    print('')
    samples = np.row_stack(samples).ravel(order='C').reshape(n, -1)
    l1_norm = np.linalg.norm(samples, ord=1, axis=1) + 2**-23
    return np.sign(samples) * np.sqrt(np.abs(samples) / l1_norm.reshape(-1, 1))


def compute_features(imfile):
    im = cv2.imread(imfile, cv2.IMREAD_GRAYSCALE)
    kp = DETECTOR.detect(im)
    kp, desc = DESCRIPTOR.compute(im, kp)
    l1_norm = np.linalg.norm(desc, ord=1, axis=1) + 2**-23
    desc = np.sign(desc) * np.sqrt(np.abs(desc) / l1_norm.reshape(-1, 1))
    return {'kp': kp2arr(kp), 'desc': desc}
