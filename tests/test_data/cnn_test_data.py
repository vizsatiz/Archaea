import lasagne

CONV_NN_PARAMETERS = {

    'layers': [{
        'layer_name': 'input',
        'layer_type': 'InputLayer',
        'shape': (None, 1, 28, 28)
    }, {
        'layer_name': 'conv2d1',
        'layer_type': 'Conv2DLayer',
        'num_filters': 32,
        'filter_size': (5, 5),
        'layer_nonlinearity': 'rectify',
        'conv_window': 'glorotuniform'
    }, {
        'layer_name': 'maxpool1',
        'layer_type': 'MaxPool2DLayer',
        'pool_size': (2, 2)
    }, {
        'layer_name': 'conv2d2',
        'layer_type': 'Conv2DLayer',
        'num_filters': 32,
        'filter_size': (5, 5),
        'layer_nonlinearity': 'rectify',
        'conv_window': None
    }, {
        'layer_name': 'maxpool2',
        'layer_type': 'MaxPool2DLayer',
        'pool_size': (2, 2)
    }, {
        'layer_name': 'dropout1',
        'layer_type': 'DropoutLayer',
        'dropout_pivot': 0.5
    }, {
        'layer_name': 'dense',
        'layer_type': 'DenseLayer',
        'num_units': 256,
        'layer_nonlinearity': 'rectify'
    }, {
        'layer_name': 'dropout2',
        'layer_type': 'DropoutLayer',
        'dropout_pivot': 0.5
    }, {
        'layer_name': 'output',
        'layer_type': 'DenseLayer',
        'num_units': 10,
        'layer_nonlinearity': 'softmax'
    }],
    'update': 'nesterov_momentum',
    'update_learning_rate': 0.01,
    'update_momentum': 0.9,
    'max_epochs': 1,
    'verbose': 1

}
