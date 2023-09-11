import numpy as np
import mlp

import random


data = []
testdata = []
y = open('../AS2/train/y_train.txt')
x = open('../AS2/train/X_train.txt')
#xtest = open('test/X_test.txt')
#ytest = open('test/y_test.txt')

for _ in range(7352):
    data.append((np.fromstring(x.readline().strip(), dtype=np.float32, sep=' '), int(y.readline().strip())))

#for _ in range(2947):
    #testdata.append((np.fromstring(xtest.readline().strip(), dtype=np.float32, sep=' '), int(ytest.readline().strip())))

y.close()
x.close()
#ytest.close()
#xtest.close()

random.shuffle(data)

train = 200
valid = 100
test = 200

train_in = data[:train]
del data[:train]
valid_in = data[:valid]
del data[:valid]
test_in = data[:test]
del data[:test]

train_tgt = np.zeros((train, 6))
valid_tgt = np.zeros((valid, 6))
test_tgt = np.zeros((train, 6))

for i in range(train):
    train_tgt[i][train_in[i][1]-1] = 1
    train_in[i] = train_in[i][0]

for i in range(test):
    test_tgt[i][test_in[i][1] - 1] = 1
    test_in[i] = test_in[i][0]

for i in range(valid):
    valid_tgt[i][valid_in[i][1] - 1] = 1
    valid_in[i] = valid_in[i][0]

train_in = np.asarray(train_in)
test_in = np.asarray(test_in)
valid_in = np.asarray(valid_in)

#results = np.array([(10, 0)])
#for idx, i in np.ndenumerate(results[:, 0]):
#    print("----- " + str(i))
#    net = mlp.mlp(train_in, train_tgt, i, outtype='softmax')
#    net.mlptrain(train_in, train_tgt, 0.25, 100)
#    net.earlystopping(train_in, train_tgt, valid_in, valid_tgt, 0.1)
#   results[idx, 1] = net.confmat(test_in, test_tgt)



net = mlp.mlp(train_in, train_tgt, 10, outtype='softmax')
net.mlptrain(train_in, train_tgt, 0.25, 10)
net.earlystopping(train_in, train_tgt, valid_in, valid_tgt, 0.1)
net.confmat(test_in, test_tgt)


