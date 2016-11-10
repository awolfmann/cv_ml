from __future__ import print_function
from __future__ import division

import sys

from os import listdir, makedirs
from os.path import join, splitext, abspath, split, exists
from multiprocessing import Pool

import numpy as np
np.seterr(all='raise')
# np.seterr(divide='ignore', invalid='ignore')

import cv2

from utils import (load_data, save_data, load_index, save_index, 
    get_random_sample, compute_features, arr2kp)

from sklearn.cluster import KMeans

from scipy.spatial import distance
from skimage.measure import ransac
from skimage.transform import AffineTransform

import base64


N_QUERY = 100

def read_image_list(imlist_file):
    return [f.rstrip() for f in open(imlist_file)]

def pca_fit(samples):
    _, ndim = samples.shape
    P = np.empty((ndim, ndim))
    mu = np.empty(ndim)
    '''
    1) computar la media (mu) de las muestras de entrenamiento
    2) computar matriz de covarianza
    3) computar autovectores y autovalores de la matriz de covarianza
    4) ordenar autovectores por valores decrecientes de los autovalores
    '''
    # np.mean por columna de la matriz samples, con las mean armar vector mu
    # mu = samples.mean(axis=0) # to take the mean of each col
    # P = np.cov(samples)
    # val, vec = np.linalg.eigh(P)  # para calcular autovectores y autovalores
    # idxs = np.argsort(-val) # ordenar autovectores por autovalores decrecientes
    # P = [vec[i] for i in idxs]
    
    return P, mu


def pca_project(x, P, mu, dim):
    # print('x.shape',x.shape)
    # print('mu.shape',mu.shape)
    # print((x - mu).reshape(1, -1))
    # print('P[:, :dim].shape', P[:, :dim].shape)
    return np.dot((x - mu).reshape(1, -1), P[:, :dim]).squeeze()



    # P, mu  = pca_fit(samples)
    # pca_samples = []
    # for sample in samples:
    #     pca_sample = pca_project(x, P, mu, 16) # 16, 32
    #     pca_samples.append(pca_sample)
    # compute vocabulary

def l2norm_feature(feature):
    l12_norm = np.linalg.norm(feature, ord=2) + 2**-23
    feature = feature / l12_norm
    return feature


if __name__ == "__main__":
    random_state = np.random.RandomState(12345)

    # ----------------
    # BUILD VOCABULARY
    # ----------------

    unsup_base_path = '/home/ariel/Documents/cv_ml/lab2_v0.2/ukbench/full/'
    ## CORREGIR PATH
    unsup_image_list_file = 'image_list.txt'

    input_path = 'cache'
    output_path = 'cache16'
    l2_output_path = 'cache16_l2'
    output_path32 = 'cache32'
    l2_output_path32 = 'cache32_l2'

    # compute random samples
    n_samples = int(1e5)
    # cambiar directorio a cache16
    unsup_samples_file_original = join(input_path, 'samples_{:d}.dat'.format(n_samples))
    unsup_samples_file_16 = join(output_path, 'samples_{:d}.dat'.format(n_samples))
    unsup_samples_file_16_l2 = join(l2_output_path, 'samples_{:d}.dat'.format(n_samples))
    unsup_samples_file_32 = join(output_path32, 'samples_{:d}.dat'.format(n_samples))
    unsup_samples_file_32_l2 = join(l2_output_path32, 'samples_{:d}.dat'.format(n_samples))
        
    P = None
    mu = None
    # PCA 16 projection
    if not exists(unsup_samples_file_16):
        print(1)
        unsup_samples_original = get_random_sample(read_image_list(unsup_image_list_file),
                                          unsup_base_path, n_samples=n_samples,
                                          random_state=random_state)
        # pca_fit y pca_project con unsup_samples
        P, mu = pca_fit(unsup_samples_original)
        unsup_samples_16 = []
        for sample in unsup_samples_original:
            pca_sample = pca_project(sample, P, mu, 16) # 16, 32
            unsup_samples_16.append(pca_sample)
        
        save_data(unsup_samples_16, unsup_samples_file_16)
        print('{} saved'.format(unsup_samples_file_16))
        print('PCA 16 projection')

    # PCA 16 projection L2 Norm
    if not exists(unsup_samples_file_16_l2):
        print(2)
        unsup_samples_16_l2 = []
        unsup_samples_16 = load_data(unsup_samples_file_16)
        P, mu = pca_fit(unsup_samples_16)
        for feature in unsup_samples_16:
            l2_feat = l2norm_feature(feature)
            pca_sample_l2 = pca_project(l2_feat, P, mu, 16)
            unsup_samples_16_l2.append(pca_sample_l2)
        
        save_data(unsup_samples_16_l2, unsup_samples_file_16_l2)
        print('{} saved'.format(unsup_samples_file_16_l2))
        print('PCA 16 projection l2 norm')

    # PCA 32 projection
    if not exists(unsup_samples_file_32):
        unsup_samples_original = get_random_sample(read_image_list(unsup_image_list_file),
                                          unsup_base_path, n_samples=n_samples,
                                          random_state=random_state)
        # pca_fit y pca_project con unsup_samples
        P, mu = pca_fit(unsup_samples_original)
        unsup_samples_32 = []
        for sample in unsup_samples_original:
            pca_sample = pca_project(sample, P, mu, 32) # 16, 32
            unsup_samples_32.append(pca_sample)
        
        save_data(unsup_samples_32, unsup_samples_file_32)
        print('{} saved'.format(unsup_samples_file_32))
        print('computed pca 32')

    # PCA 32 projection L2 Norm
    if not exists(unsup_samples_file_32_l2):
        
        unsup_samples_32_l2 = []
        unsup_samples_32 = load_data(unsup_samples_file_32)
        P, mu = pca_fit(unsup_samples_32)
        for feature in unsup_samples_32:
            l2_feat = l2norm_feature(feature)
            pca_sample_l2 = pca_project(l2_feat, P, mu, 32)
            unsup_samples_32_l2.append(pca_sample_l2)
        
        save_data(unsup_samples_32_l2, unsup_samples_file_32_l2)
        print('{} saved'.format(unsup_samples_file_32_l2))
        print('PCA 32 l2 norm')
