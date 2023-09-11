import numpy as np
import mlp
import random


def fitness(population):

    output = []

    y = open('../AS2/train/y_train.txt')
    x = open('../AS2/train/X_train.txt')
    data = []
    for _ in range(7352):
        data.append((np.fromstring(x.readline().strip(), dtype=np.float32, sep=' '), int(y.readline().strip())))
    y.close()
    x.close()

    random.shuffle(data)
    train = 500
    valid = 100
    test = train

    train_in = data[:train]
    del data[:train]
    valid_in = data[:valid]
    del data[:valid]
    test_in = data[:test]
    del data[:test]

    train_tgt = np.zeros((train, 6))
    valid_tgt = np.zeros((valid, 6))
    test_tgt = np.zeros((test, 6))

    for i in range(train):
        train_tgt[i][train_in[i][1] - 1] = 1
        train_in[i] = train_in[i][0]

    for i in range(test):
        test_tgt[i][test_in[i][1] - 1] = 1
        test_in[i] = test_in[i][0]

    for i in range(valid):
        valid_tgt[i][valid_in[i][1] - 1] = 1
        valid_in[i] = valid_in[i][0]

    for n in range(len(population)):
        output.append(0)
        temp_train = train_in.copy()
        temp_test = test_in.copy()
        temp_valid = valid_in.copy()
        removed = 0

        for m in range(561):
            if population[n][m] == 0:
                for x in range(len(temp_train)):
                    temp_train[x] = np.delete(temp_train[x], m-removed)
                    temp_test[x] = np.delete(temp_test[x], m-removed)
                for x in range(len(temp_valid)):
                    temp_valid[x] = np.delete(temp_valid[x], m-removed)
                removed += 1

        temp_train = np.asarray(temp_train)
        temp_valid = np.asarray(temp_valid)
        temp_test = np.asarray(temp_test)

        net = mlp.mlp(temp_train, train_tgt, 10, outtype='softmax')
        net.mlptrain(temp_train, train_tgt, 0.25, 100)
        net.earlystopping(temp_train, train_tgt, temp_valid, valid_tgt, 0.1)
        score = net.confmat(temp_test, test_tgt) # between 80-98 usually
        if score < 80:
            output[n] = 0
        else:
            output[n] = (score-80) * removed # fitness based on how far score went over 80 * by how many genes/features were removed

    return np.asarray(output)
