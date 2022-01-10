import os
import numpy as np

_items_pop = None


def countPop(dataset_name='ml-1m', datafolder='Data'):
    testfile_name = dataset_name + '.test.rating'
    trainfile_name = dataset_name + '.train.rating'
    dataset_test_raw = np.genfromtxt(os.path.join(datafolder, testfile_name), dtype='int', delimiter='\t')
    dataset_train_raw = np.genfromtxt(os.path.join(datafolder, trainfile_name), dtype='int', delimiter='\t')
    dataset_raw = np.concatenate((dataset_test_raw, dataset_train_raw), axis=0)
    users_num = float(len(np.unique(dataset_raw[:, 0])))
    (itemIDs, interactions_counts) = np.unique(dataset_raw[:, 1], return_counts=True)
    normalized_interactions_counts = interactions_counts.astype(float)/users_num
    items_pop = dict(zip(itemIDs, normalized_interactions_counts))
    return items_pop


def getItemsFeatures(filename='movies.dat', datafolder='Data'):
    itemIDs = []
    items_features = []
    with open(os.path.join(datafolder, filename)) as file:
        for line in file:
            (itemID, _, item_features) = line.split('::')
            itemIDs.append(int(itemID))
            items_features.append(item_features.replace('\n', '').split('|'))
    unique_features = np.unique(np.concatenate(items_features).ravel())
    for i, item_features in enumerate(items_features):
        items_features[i] = np.in1d(unique_features, item_features).astype(int)
    return dict(zip(itemIDs, items_features))


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


if __name__ == '__main__':
    getUsersHistory()
    # getItemsFeatures()
    # items_pop = countPop('ml-1m')
    # print(items_pop)
