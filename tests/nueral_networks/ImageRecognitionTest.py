import os
from numpy import ravel

from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from pybrain.tools.xml.networkwriter import NetworkWriter
from pybrain.tools.xml.networkreader import NetworkReader
import machine_learning.nueral_network.simple_neural_network as snn
import machine_learning.model.NetworkArchitecture as netArch
import machine_learning.nueral_network.nn_trainer_factory as trainFact
import machine_learning.common_utils.common_constants as constants
from sklearn import datasets

olivetti = datasets.fetch_olivetti_faces()
X, y = olivetti.data, olivetti.target
ds = ClassificationDataSet(4096, 1 , nb_classes=40)
for k in xrange(len(X)):
  ds.addSample(ravel(X[k]),y[k])
tstdata, trndata = ds.splitWithProportion( 0.25 )
#print tstdata;
trndata._convertToOneOfMany( )
tstdata._convertToOneOfMany( )
#print tstdata;
#fnn = buildNetwork( trndata.indim, 64 , trndata.outdim, outclass=SoftmaxLayer )
#if  os.path.isfile('oliv.xml'):
 #fnn = NetworkReader.readFrom('oliv.xml')
#else:
dimension = [trndata.indim, 64, trndata.outdim]
fnn = snn.SimpleNeuralNetwork(netArch.NetworkArchitecture(dimension)).get_simple_neural_network()

    #buildNetwork(trndata.indim, 32, 64, trndata.outdim, outclass=SoftmaxLayer )
#print fnn;
 #trainer = BackpropTrainer(fnn, dataset=trndata, momentum=0.1, learningrate=0.01 , verbose=True, weightdecay=0.01)
NetworkWriter.writeToFile(fnn, 'oliv.xml')
parameters = {'network' : fnn, 'dataset': trndata, 'momentum': 0.1, 'learningrate': 0.01, 'verbose': True, 'weightdecay': 0.01}
trainer = trainFact.NetworkTrainer(parameters).get_ann_trainer(constants.BACK_PROP_TRAINER)
trainer.train(50)


#trainer.trainEpochs (50)
print 'Percent Error on Test dataset: ' , trainer.percentage_error_on_dataset(tstdata)
