import os
import gzip
import unittest
import numpy as np
import cPickle as pickle
import matplotlib.cm as cm
from urllib import urlretrieve
import matplotlib.pyplot as plt
import machine_learning.conv_nn.network_trainer as trainer
import machine_learning.conv_nn.network_builder as builder
import machine_learning_tests.test_data.cnn_test_data as constants
import machine_learning.model_persistance.lasagne_model as persistor


class LasagnePersistenceHelperTests(unittest.TestCase):

    def lasagne_cnn_model_persistance_tests(self):

        def load_dataset():
            url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
            filename = 'mnist.pkl.gz'
            if not os.path.exists(filename):
                print("Downloading MNIST dataset...")
                urlretrieve(url, filename)
            with gzip.open(filename, 'rb') as f:
                data = pickle.load(f)
            X_train, y_train = data[0]
            X_val, y_val = data[1]
            X_test, y_test = data[2]
            X_train = X_train.reshape((-1, 1, 28, 28))
            X_val = X_val.reshape((-1, 1, 28, 28))
            X_test = X_test.reshape((-1, 1, 28, 28))
            y_train = y_train.astype(np.uint8)
            y_val = y_val.astype(np.uint8)
            y_test = y_test.astype(np.uint8)
            return X_train, y_train, X_val, y_val, X_test, y_test
        X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
        plt.imshow(X_train[0][0], cmap=cm.binary)
        net1 = builder.ConvNetworkBuilder(constants.CONV_NN_PARAMETERS).build()
        cnn_trainer = trainer.ConvNeuralNetTrainer(net1)
        # Train the network
        nn = cnn_trainer.train(X_train, y_train)
        preds = cnn_trainer.predict(X_test)
        [report, confusion_matric] = cnn_trainer.confusion_matrix(X_test, y_test)
        state = persistor.LasagneModelPersistenceHelper.get_model_state(cnn_trainer.network)
        new_cnn_net = builder.ConvNetworkBuilder(constants.CONV_NN_PARAMETERS).build()
        new_cnn_net_trained = persistor.LasagneModelPersistenceHelper.initialize_model_with_state(new_cnn_net, state)
        cnn_trainer_new = trainer.ConvNeuralNetTrainer(new_cnn_net_trained)
        [report_new, confusion_matric_new] = cnn_trainer.confusion_matrix(X_test, y_test)
        self.assertEqual(confusion_matric_new.item(1), confusion_matric.item(1))

