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

import os
# from sklearn.metrics import jaccard_score, hamming_loss

# from numba import jit, autojit

# Global variables that are shared across processes
_model = None
_testRatings = None
_testNegatives = None
_K = None
_items_pop = None
_items_features = None
_similarity = None
_users_history = None


def evaluate_model(model, testRatings, testNegatives, K, num_thread, items_pop, items_features, users_history, similarity='jaccard'):
    """
    Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
    Return: score of each test rating.
    """
    global _model
    global _testRatings
    global _testNegatives
    global _K
    global _items_pop
    global _items_features
    global _similarity
    global _users_history
    _model = model
    _testRatings = testRatings
    _testNegatives = testNegatives
    _K = K
    _items_pop = items_pop
    _items_features = items_features
    _similarity = similarity
    _users_history = users_history

    hits, ndcgs, novelties, expectednesses, IDLs, unserendipities = [], [], [], [], [], []
    if (num_thread > 1):  # Multi-thread
        pool = multiprocessing.Pool(processes=num_thread)
        res = pool.map(eval_one_rating, range(len(_testRatings)))
        pool.close()
        pool.join()
        hits = [r[0] for r in res]
        ndcgs = [r[1] for r in res]
        novelties = [r[2] for r in res]
        expectednesses = [r[3] for r in res]
        IDLs = [r[4] for r in res]
        unserendipities = [r[5] for r in res]
        return (hits, ndcgs, novelties, expectednesses, IDLs, unserendipities)
    # Single thread
    for idx in xrange(len(_testRatings)):
        (hr, ndcg, novelty, expectedness, IDL, unserendipity) = eval_one_rating(idx)
        hits.append(hr)
        ndcgs.append(ndcg)
        novelties.append(novelty)
        expectednesses.append(expectedness)
        IDLs.append(IDL)
        unserendipities.append(unserendipity)
    return (hits, ndcgs, novelties, expectednesses, IDLs, unserendipities)


def eval_one_rating(idx):
    rating = _testRatings[idx]
    items = _testNegatives[idx]
    u = rating[0]
    gtItem = rating[1]
    items.append(gtItem)
    # Get prediction scores
    map_item_score = {}
    users = np.full(len(items), u, dtype='int32')
    predictions = _model.predict([users, np.array(items)],
                                 batch_size=100, verbose=0)
    for i in xrange(len(items)):
        item = items[i]
        map_item_score[item] = predictions[i]
    items.pop()

    # Evaluate top rank list
    ranklist = heapq.nlargest(_K, map_item_score, key=map_item_score.get)
    hr = getHitRatio(ranklist, gtItem)
    ndcg = getNDCG(ranklist, gtItem)
    novelty = getNovelty(ranklist)
    expectedness = getExpectedness(ranklist)
    IDL = getILD(ranklist, _similarity)
    unserendipity = getUnserendipity(ranklist, u, _similarity)

    # if not idx == 0:
    #     print('------------------------------------')
    #     print('idx', idx)
    #     print('ranklist', ranklist)
    #     print('ranklist_type', type(ranklist))
    #     print('gtItem', gtItem)
    #     print('gtItem_type', type(gtItem))
    #     print('hr', hr)
    #     print('ndcg', ndcg)
    #     print('novelty', novelty)
    #     print('expectedness', expectedness)
    #     print('ILD', IDL)
    #     print('unserendipity', unserendipity)
    return (hr, ndcg, novelty, expectedness, IDL, unserendipity)


def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0


def getNDCG(ranklist, gtItem):
    for i in xrange(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i + 2)
    return 0


def getNovelty(ranklist):
    nominator = 0
    for i in xrange(len(ranklist)):
        item = ranklist[i]
        nominator += np.log2(_items_pop[item])
    return nominator / _K


def getExpectedness(ranklist):
    nominator = 0
    for i in xrange(len(ranklist)):
        item = ranklist[i]
        nominator += _items_pop[item]
    return nominator / _K


def getILD(ranklist, similarity='jaccard'):
    items_features_keys = _items_features.keys()
    nominator = 0
    for i in xrange(len(ranklist)):
        for j in xrange(len(ranklist)):
            if i == j: continue
            item_i = ranklist[i]
            item_j = ranklist[j]
            if item_i in items_features_keys and item_j in items_features_keys:
                if similarity == 'jaccard':
                    nominator += 1 - getJaccardSimilarity(_items_features[item_i], _items_features[item_j])
                else:
                    nominator += 1 - getHammingSimilarity(_items_features[item_i], _items_features[item_j])
    return nominator / (len(ranklist)*(len(ranklist) - 1))


def getUnserendipity(ranklist, userID, similarity='jaccard'):
    items_features_keys = _items_features.keys()
    if len(_users_history[userID]) == 0:
        return 0
    nominator = 0
    for history_item in _users_history[userID]:
        second_nominator = 0
        for i in xrange(len(ranklist)):
            item = ranklist[i]
            if item in items_features_keys and history_item in items_features_keys:
                if similarity == 'jaccard':
                    second_nominator += getJaccardSimilarity(_items_features[item], _items_features[history_item])
                else:
                    second_nominator += getHammingSimilarity(_items_features[item], _items_features[history_item])
        nominator += second_nominator / _K
    return nominator / len(_users_history[userID])


def getUsersHistory(dataset_name='ml-1m', datafolder='Data'):
    trainfile_name = dataset_name + '.train.rating'
    dataset_raw = np.genfromtxt(os.path.join(datafolder, trainfile_name), dtype='int', delimiter='\t')
    usersIDs = dataset_raw[:, 0]
    itemsIDs = dataset_raw[:, 1]
    users_history = dict()
    for userID, itemID in zip(usersIDs, itemsIDs):
        if userID in users_history.keys():
            users_history[userID].append(itemID)
        else:
            users_history[userID] = [itemID]
    for userID in users_history.keys():
        users_history[userID] = np.array(list(set(users_history[userID])))
    return users_history


def getPop(dataset_name='ml-1m', datafolder='Data'):
    testfile_name = dataset_name + '.test.rating'
    trainfile_name = dataset_name + '.train.rating'
    dataset_test_raw = np.genfromtxt(os.path.join(datafolder, testfile_name), dtype='int', delimiter='\t')
    dataset_train_raw = np.genfromtxt(os.path.join(datafolder, trainfile_name), dtype='int', delimiter='\t')
    dataset_raw = np.concatenate((dataset_test_raw, dataset_train_raw), axis=0)
    users_num = float(len(np.unique(dataset_raw[:, 0])))
    (itemIDs, interactions_counts) = np.unique(dataset_raw[:, 1], return_counts=True)
    normalized_interactions_counts = interactions_counts.astype(float) / users_num
    items_pop = dict(zip(itemIDs, normalized_interactions_counts))
    return items_pop


def getItemsFeatures(filename='movies.dat', datafolder='Data'):
    itemIDs = []
    items_features = []
    with open(os.path.join(datafolder, filename)) as file:
        for line in file:
            (itemID, _, item_features) = line.split('::')
            itemIDs.append(int(itemID) - 1)
            items_features.append(item_features.replace('\n', '').split('|'))
    unique_features = np.unique(np.concatenate(items_features).ravel())
    for i, item_features in enumerate(items_features):
        items_features[i] = np.in1d(unique_features, item_features).astype(int)
    return dict(zip(itemIDs, items_features))


def getJaccardSimilarity(array_a, array_b):
    assert len(array_a) == len(array_b)
    array_a = array_a * np.arange(1, len(array_a) + 1)
    list2 = array_b * np.arange(1, len(array_b) + 1)
    s1 = set(array_a)
    s2 = set(list2)
    return float(len(s1.intersection(s2)) / len(s1.union(s2)))
    # return jaccard_score(array_a, array_b)


def getHammingSimilarity(array_a, array_b):
    assert len(array_a) == len(array_b)
    return 1 - sum(c1 != c2 for c1, c2 in zip(array_a, array_b))/len(array_a)
    # return 1.0 - hamming_loss(array_a, array_b)
