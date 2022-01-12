'''
Created on Apr 15, 2016
Modified on Jan 9, 2021
Evaluate the performance of Top-K recommendation:
    Protocol: leave-1-out evaluation
    Measures: Hit Ratio, NDCG, novelty, expectedness, IDL and unserendipity
    (more details are in: Xiangnan He, et al. Fast Matrix Factorization for Online Recommendation with Implicit Feedback. SIGIR'16)

@author: hexiangnan
@modifications author: Sebastian Lewalski
'''
import math
import heapq  # for retrieval topK
import multiprocessing
import numpy as np
from time import time

from keras.optimizers import Adam
import GMF, MLP
from NeuMF import load_pretrain_model, get_model
from Dataset import Dataset
import os
import sys
from evaluate import *
import multiprocessing as mp
# from sklearn.metrics import jaccard_score, hamming_loss

# from numba import jit, autojit


if __name__ == '__main__':
    # num_epochs = args.epochs
    # batch_size = args.batch_size
    # mf_dim = args.num_factors
    # layers = eval(args.layers)
    # reg_mf = args.reg_mf
    # reg_layers = eval(args.reg_layers)
    # num_negatives = args.num_neg
    # learning_rate = args.lr
    # learner = args.learner
    # verbose = args.verbose
    # mf_pretrain = args.mf_pretrain
    # mlp_pretrain = args.mlp_pretrain

    print("I'm alive")
    print('Python', sys.version)
    print('Numpy', np.__version__)

    path = 'Data/'
    dataset_nam = 'ml-1m'
    layers_args = '[64,32,16,8]'
    layers = eval(layers_args)
    mf_dim = 8
    reg_layers_arg = '[0,0,0,0]'
    reg_layers = eval(reg_layers_arg)
    reg_mf = 0.0
    learning_rate = 0.001
    mf_pretrain = 'Pretrain/ml-1m_GMF_8_1501651698.h5'
    mlp_pretrain = 'Pretrain/ml-1m_MLP_[64,32,16,8]_1501652038.h5'

    similarity_arg = 'jaccard'

    topK = 10
    evaluation_threads = mp.cpu_count()
    model_out_file = 'Pretrain/%s_NeuMF_%d_%s_%d.h5' % (dataset_nam, mf_dim, layers_args, time())

    # Loading data
    t1 = time()
    dataset = Dataset(path + dataset_nam)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_items = train.shape
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d"
          % (time() - t1, num_users, num_items, train.nnz, len(testRatings)))

    model = get_model(num_users, num_items, mf_dim, layers, reg_layers, reg_mf)
    model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')

    gmf_model = GMF.get_model(num_users, num_items, mf_dim)
    gmf_model.load_weights(mf_pretrain)
    mlp_model = MLP.get_model(num_users, num_items, layers, reg_layers)
    mlp_model.load_weights(mlp_pretrain)
    model = load_pretrain_model(model, gmf_model, mlp_model, len(layers))
    print("Load pretrained GMF (%s) and MLP (%s) models done. " % (mf_pretrain, mlp_pretrain))

    print('Start evaluation')
    items_pop = getPop(dataset_nam)
    users_history = getUsersHistory(dataset_nam)
    items_features = getItemsFeatures()
    (hits, ndcgs, novelty, expectedness, IDL, unserendipities) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads,
                                                          items_pop, items_features, users_history, similarity=similarity_arg)
    hr, ndcg, novelty, expectedness, IDL, unserendipity = np.array(hits).mean(), np.array(ndcgs).mean(), np.array(novelty).mean(), \
                                           np.array(expectedness).mean(), np.array(IDL).mean(), np.array(unserendipities).mean()
    print('Result: HR = %.4f, NDCG = %.4f, novelty = %.10f, expectedness = %.10f, IDL = %.10f, unserendipity = %.10f' % (hr, ndcg, novelty, expectedness, IDL, unserendipity))
