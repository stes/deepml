"""
ResNet-50, ResNet-101 and ResNet-152 models

from the paper: "Deep Residual Learning for Image Recognition", He et al. (2015)
[arXiv:1512.03385]
https://github.com/KaimingHe/deep-residual-networks (Shaoqing Ren)
License: see https://github.com/KaimingHe/deep-residual-networks/blob/master/LICENSE

Further References
------------------

Based on visualization from

ResNet 50: http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006
ResNet 101: http://ethereon.github.io/netscope/#/gist/b21e2aae116dc1ac7b50
ResNet 152: http://ethereon.github.io/netscope/#/gist/d38f3e6091952b45198b

Weights from
------------

http://www.vlfeat.org/matconvnet/pretrained/

Code only slightly adapted, original one from
https://github.com/Lasagne/Recipes
"""

import lasagne as nn
from lasagne.layers import InputLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import BatchNormLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import GlobalPoolLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import ElemwiseSumLayer
from lasagne.layers import DenseLayer
from lasagne.nonlinearities import rectify, softmax

import os
import logging
import h5py
import collections
import theano


### -----------------------------------------------------------------------------
### I/O

def load_weights(fname, net):
    log = logging.getLogger(__name__)
    with h5py.File(fname, "r") as ds:
        for key in net.keys():
            try:
                #print("Entering group", key)
                grp = ds[key]
                for param in net[key].params.keys():
                    p = grp[str(param)]

                    param_shape = param.get_value(borrow=True).shape
                    value_shape = p.shape

                    #print("Adapting:", key, param)
                    param.set_value(p[...])
                    assert (param.get_value() == p[...]).all()
                    #print("\tSuccess.")
            except:
                pass
                #log.exception("Error when adapting weights for key " + key)
                #print("ERROR when adapting weights: ", key)

def save_weights(net, savedir, epoch=-1):
    """ Save weights in the given dir, ordered by epoch
    final name will be [savedir]/weights/weights-ep[epoch].hdf5
    """
    weight_dir = os.path.join(savedir, "weights")
    if not os.path.exists(weight_dir):
        os.mkdir(weight_dir)
    if epoch == -1:
        fullname = os.path.join(weight_dir, "weights.hdf5")
        if os.path.exists(fullname):
            os.unlink(fullname)
    else:
        fullname = os.path.join(weight_dir, "weights-ep" + str(epoch) + ".hdf5")

    with h5py.File(fullname) as f:
        for key in list(net.keys()):
            for param in net[key].get_params():
                f.create_dataset(key + "/" + str(param), data=param.get_value(), dtype="float32")


### -----------------------------------------------------------------------------
### Network related functions

def build_simple_block(incoming_layer, names,
                       num_filters, filter_size, stride, pad,
                       use_bias=False, nonlin=rectify):
    """Creates stacked Lasagne layers ConvLayer -> BN -> (ReLu)
    Parameters:
    ----------
    incoming_layer : instance of Lasagne layer
        Parent layer
    names : list of string
        Names of the layers in block
    num_filters : int
        Number of filters in convolution layer
    filter_size : int
        Size of filters in convolution layer
    stride : int
        Stride of convolution layer
    pad : int
        Padding of convolution layer
    use_bias : bool
        Whether to use bias in conlovution layer
    nonlin : function
        Nonlinearity type of Nonlinearity layer
    Returns
    -------
    tuple: (net, last_layer_name)
        net : dict
            Dictionary with stacked layers
        last_layer_name : string
            Last layer name
    """
    net = []
    names = list(names)
    net.append((
            names[0],
            ConvLayer(incoming_layer, num_filters, filter_size, pad, stride,
                      flip_filters=False, nonlinearity=None) if use_bias
            else ConvLayer(incoming_layer, num_filters, filter_size, stride, pad, b=None,
                           flip_filters=False, nonlinearity=None)
        ))

    net.append((
            names[1],
            BatchNormLayer(net[-1][1])
        ))
    if nonlin is not None:
        net.append((
            names[2],
            NonlinearityLayer(net[-1][1], nonlinearity=nonlin)
        ))

    return dict(net), net[-1][0]


def build_residual_block(incoming_layer, ratio_n_filter=1.0, ratio_size=1.0, has_left_branch=False,
                         upscale_factor=4, ix=''):
    """Creates two-branch residual block
    Parameters:
    ----------
    incoming_layer : instance of Lasagne layer
        Parent layer
    ratio_n_filter : float
        Scale factor of filter bank at the input of residual block
    ratio_size : float
        Scale factor of filter size
    has_left_branch : bool
        if True, then left branch contains simple block
    upscale_factor : float
        Scale factor of filter bank at the output of residual block
    ix : int
        Id of residual block
    Returns
    -------
    tuple: (net, last_layer_name)
        net : dict
            Dictionary with stacked layers
        last_layer_name : string
            Last layer name
    """
    simple_block_name_pattern = ['res%s_branch%i%s', 'bn%s_branch%i%s', 'res%s_branch%i%s_relu']

    net = collections.OrderedDict()

    # right branch
    net_tmp, last_layer_name = build_simple_block(
        incoming_layer, map(lambda s: s % (ix, 2, 'a'), simple_block_name_pattern),
        int(nn.layers.get_output_shape(incoming_layer)[1]*ratio_n_filter), 1, int(1.0/ratio_size), 0)
    net.update(net_tmp)

    net_tmp, last_layer_name = build_simple_block(
        net[last_layer_name], map(lambda s: s % (ix, 2, 'b'), simple_block_name_pattern),
        nn.layers.get_output_shape(net[last_layer_name])[1], 3, 1, 1)
    net.update(net_tmp)

    net_tmp, last_layer_name = build_simple_block(
        net[last_layer_name], map(lambda s: s % (ix, 2, 'c'), simple_block_name_pattern),
        nn.layers.get_output_shape(net[last_layer_name])[1]*upscale_factor, 1, 1, 0,
        nonlin=None)
    net.update(net_tmp)

    right_tail = net[last_layer_name]
    left_tail = incoming_layer

    # left branch
    if has_left_branch:
        net_tmp, last_layer_name = build_simple_block(
            incoming_layer, map(lambda s: s % (ix, 1, ''), simple_block_name_pattern),
            int(nn.layers.get_output_shape(incoming_layer)[1]*4*ratio_n_filter), 1, int(1.0/ratio_size), 0,
            nonlin=None)
        net.update(net_tmp)
        left_tail = net[last_layer_name]

    net['res%s' % ix] = ElemwiseSumLayer([left_tail, right_tail], coeffs=1)
    net['res%s_relu' % ix] = NonlinearityLayer(net['res%s' % ix], nonlinearity=rectify)

    return net, 'res%s_relu' % ix

def build_encoder(input_layer, net, variant = "resnet50"):
    """ Build a feature encoder module using the ResNet-x architecture

    Parameters
    ----------
    input_layer : Lasagne layer serving the input to this network module
    net : dict (recommended is collections.OrderedDict) to collect layers
    variant : str, one of "resnet50", "resnet101", "resnet152"
    """
    assert variant in ["resnet50", "resnet101", "resnet152"], "Unsupported variant: " + variant

    structure = \
    {
        'resnet50' : 
            (
                list('abcd'),
                list('abcdef')
            ),
        'resnet101' :
            (
                list(('a',) + tuple('b{}'.format(i) for i in range(3+1))),
                list(('a',) + tuple('b{}'.format(i) for i in range(22+1)))
            ),
        'resnet152' : 
            (
                list(('a',) + tuple('b{}'.format(i) for i in range(7+1))),
                list(('a',) + tuple('b{}'.format(i) for i in range(35+1)))
            )
    }

    #BLOCK1 begins here
    sub_net, parent_layer_name = build_simple_block(
        net['input'], ['conv1', 'bn_conv1', 'conv1_relu'],
        64, 7, 3, 2, use_bias=True)
    net.update(sub_net)
    net['pool1'] = PoolLayer(net[parent_layer_name], pool_size=3, stride=2, pad=0, mode='max', ignore_border=False)
    parent_layer_name = 'pool1'

    #BLOCK2 begins here
    block_size = list('abc')
    for c in block_size:
        if c == 'a':
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1, 1, True, 4, ix='2%s' % c)
        else:
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/4, 1, False, 4, ix='2%s' % c)
        net.update(sub_net)

    #BLOCK3 begins here
    block_size = list(structure[variant][0])
    for c in block_size:
        if c == 'a':
            sub_net, parent_layer_name = build_residual_block(
                net[parent_layer_name], 1.0/2, 1.0/2, True, 4, ix='3%s' % c)
        else:
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/4, 1, False, 4, ix='3%s' % c)
        net.update(sub_net)

    #BLOCK4 begins here
    block_size = list(structure[variant][1])
    for c in block_size:
        if c == 'a':
            sub_net, parent_layer_name = build_residual_block(
                net[parent_layer_name], 1.0/2, 1.0/2, True, 4, ix='4%s' % c)
        else:
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/4, 1, False, 4, ix='4%s' % c)
        net.update(sub_net)

    #BLOCK5 begins here
    block_size = list('abc')
    for c in block_size:
        if c == 'a':
            sub_net, parent_layer_name = build_residual_block(
                net[parent_layer_name], 1.0/2, 1.0/2, True, 4, ix='5%s' % c)
        else:
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/4, 1, False, 4, ix='5%s' % c)
        net.update(sub_net)
    return net[parent_layer_name]

def build_model(input_var, input_shape=(None, 3, 224, 224), variant="resnet50"):
    """ Build a ResNet Classification Network
    """
    net = collections.OrderedDict()
    net['input'] = InputLayer(input_shape, input_var=input_var)
    l_enc = build_encoder(net["input"], net, variant=variant)
    net['pool5'] = PoolLayer(l_enc, pool_size=7, stride=1, pad=0,
                             mode='average_exc_pad', ignore_border=False)
    net['fc1000'] = DenseLayer(net['pool5'], num_units=1000, nonlinearity=None)
    
    net['prob'] = NonlinearityLayer(net['fc1000'], nonlinearity=softmax)
    
    return net

def predict_fn(variant="resnet50", weightdir=None):
    """ Directly build the prediction function for the given resnet variant
    
    Notes
    -----
    Use images in range [0, 255] as input for the network.
    """

    input_var = theano.tensor.tensor4()
    net = build_model(input_var, input_shape=(None, 3, 224, 224), variant=variant)
    if weightdir is not None: load_weights(os.path.join(weightdir, variant + ".hdf5"), net)
    pred_fn = theano.function([input_var], nn.layers.get_output(net['prob'], deterministic=True))
    return pred_fn

def feature_fn(variant="resnet50", weightdir=None):
    """ Directly build the prediction function for the given resnet variant
    
    Notes
    -----
    Use images in range [0, 255] as input for the network.
    """

    input_var = theano.tensor.tensor4()
    net = collections.OrderedDict()
    net['input'] = InputLayer((None,3,None,None), input_var=input_var)
    l_enc = build_encoder(net["input"], net, variant=variant)
    if weightdir is not None: load_weights(os.path.join(weightdir, variant + ".hdf5"), net)
    feature_fn = theano.function([input_var], nn.layers.get_output(l_enc, deterministic=True))
    return feature_fn

