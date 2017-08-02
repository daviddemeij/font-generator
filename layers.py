import tensorflow as tf
import numpy as np
import sys, os

_default_weightnorm = False
def enable_default_weightnorm():
    global _default_weightnorm
    _default_weightnorm = True

def disable_default_weightnorm():
    global _default_weightnorm
    _default_weightnorm = False

_weights_stdev = None
def set_weights_stdev(weights_stdev):
    global _weights_stdev
    _weights_stdev = weights_stdev

def unset_weights_stdev():
    global _weights_stdev
    _weights_stdev = None

FLAGS = tf.app.flags.FLAGS
try:
    tf.app.flags.DEFINE_float('bn_stats_decay_factor', 0.99,
                              "moving average decay factor for stats on batch normalization")
except :
    pass

# Get lists of parameters for the generator and discriminator or etc
def params_with_name(n):
    return [i for i in tf.global_variables() if n in i.name]

def lrelu(x, a=0.2):
    if a is None:
        raise Exception('Must give lrelu function an alpha value!')
    if a < 1e-16:
        return tf.nn.relu(x)
    else:
        return tf.maximum(x, a * x)
            
def parse_act_fn_name(act_fn, lrelu_alpha=None):
    act_dict = {
            'none'      : lambda x: x,
            'sigmoid'   : tf.nn.sigmoid,
            'tanh'      : tf.tanh,
            'relu'      : tf.nn.relu,
            'leakyrelu' : lambda x: lrelu(x, lrelu_alpha),
            'lrelu'     : lambda x: lrelu(x, lrelu_alpha)
               }
    try:
        return act_dict[act_fn]
    except KeyError:
        raise Exception(
            'act_fn must be one of {' + ','.join(act_dict.keys()) + '}')

def bn(x, dim, is_training=True, update_batch_stats=True, collections=None, 
       name="bn", use_gamma=True):
    params_shape = (dim,)
    n = tf.to_float(tf.reduce_prod(tf.shape(x)[:-1]))
    axis = list(range(int(tf.shape(x).get_shape().as_list()[0]) - 1))
    mean = tf.reduce_mean(x, axis)
    var = tf.reduce_mean(tf.pow(x - mean, 2.0), axis)
    avg_mean = tf.get_variable(
        name=name + "_mean",
        shape=params_shape,
        initializer=tf.constant_initializer(0.0),
        collections=collections,
        trainable=False
    )

    avg_var = tf.get_variable(
        name=name + "_var",
        shape=params_shape,
        initializer=tf.constant_initializer(1.0),
        collections=collections,
        trainable=False
    )

    if use_gamma:
        gamma = tf.get_variable(
            name=name + "_gamma",
            shape=params_shape,
            initializer=tf.constant_initializer(1.0),
            collections=collections
        )
    else:
        gamma = tf.constant(1.0, dtype=tf.float32, name='gamma_constant_1')

    beta = tf.get_variable(
        name=name + "_beta",
        shape=params_shape,
        initializer=tf.constant_initializer(0.0),
        collections=collections,
    )

    if is_training:
        avg_mean_assign_op = tf.no_op()
        avg_var_assign_op = tf.no_op()
        if update_batch_stats:
            avg_mean_assign_op = tf.assign(
                avg_mean,
                FLAGS.bn_stats_decay_factor * avg_mean + (1 - FLAGS.bn_stats_decay_factor) * mean)
            avg_var_assign_op = tf.assign(
                avg_var,
                FLAGS.bn_stats_decay_factor * avg_var + (n / (n - 1))
                * (1 - FLAGS.bn_stats_decay_factor) * var)

        with tf.control_dependencies([avg_mean_assign_op, avg_var_assign_op]):
            z = (x - mean) / tf.sqrt(1e-6 + var)
    else:
        z = (x - avg_mean) / tf.sqrt(1e-6 + avg_var)

    return gamma * z + beta

def fc(x, dim_in, dim_out, seed=None, name='fc', bias=True):
    num_units_in = dim_in
    num_units_out = dim_out
    weights_initializer = tf.contrib.layers.variance_scaling_initializer(seed=seed, mode="FAN_AVG")

    weights = tf.get_variable(name + '_W',
                            shape=[num_units_in, num_units_out],
                            initializer=weights_initializer)
    if bias:
        biases = tf.get_variable(name + '_b',
                                 shape=[num_units_out],
                                 initializer=tf.constant_initializer(0.0))
        x = tf.nn.xw_plus_b(x, weights, biases)
    else:
        x = tf.matmul(x, weights)
    return x


def fc_v2(inputs,
          input_dim, 
          output_dim, 
          name, 
          rng,
          biases=True,
          init=None,
          weightnorm=None,
          gain=1.
          ):
    """
    init: None, `lecun`, 'glorot', `he`, 'glorot_he', `orthogonal`, `("uniform", range)`
    """
    #with tf.name_scope(name) as scope:

    def uniform(stdev, size):
        if _weights_stdev is not None:
            stdev = _weights_stdev
        return rng.uniform(
            low=-stdev * np.sqrt(3),
            high=stdev * np.sqrt(3),
            size=size
        ).astype('float32')

    if init == 'lecun':# and input_dim != output_dim):
        # disabling orth. init for now because it's too slow
        weight_values = uniform(
            np.sqrt(1./input_dim),
            (input_dim, output_dim)
        )

    elif init == 'glorot' or (init is None):

        weight_values = uniform(
            np.sqrt(2./(input_dim+output_dim)),
            (input_dim, output_dim)
        )

    elif init == 'he':

        weight_values = uniform(
            np.sqrt(2./input_dim),
            (input_dim, output_dim)
        )

    elif init == 'glorot_he':

        weight_values = uniform(
            np.sqrt(4./(input_dim+output_dim)),
            (input_dim, output_dim)
        )

    elif init == 'orthogonal' or \
        (init == None and input_dim == output_dim):
        
        # From lasagne
        def sample(shape):
            if len(shape) < 2:
                raise RuntimeError("Only shapes of length 2 or more are "
                                   "supported.")
            flat_shape = (shape[0], np.prod(shape[1:]))
             # TODO: why normal and not uniform?
            a = rng.normal(0.0, 1.0, flat_shape)
            u, _, v = np.linalg.svd(a, full_matrices=False)
            # pick the one with the correct shape
            q = u if u.shape == flat_shape else v
            q = q.reshape(shape)
            return q.astype('float32')
        weight_values = sample((input_dim, output_dim))
    
    elif init[0] == 'uniform':
    
        weight_values = rng.uniform(
            low=-init[1],
            high=init[1],
            size=(input_dim, output_dim)
        ).astype('float32')

    else:

        raise Exception('Invalid initialization!')

    weight_values *= gain

    weight = tf.get_variable(name + '_W', 
                             initializer=tf.constant(weight_values))

    if weightnorm==None:
        weightnorm = _default_weightnorm
    if weightnorm:
        norm_values = np.sqrt(np.sum(np.square(weight_values), axis=0))
        # norm_values = np.linalg.norm(weight_values, axis=0)

        target_norms = tf.get_variable(name + '.g',
                                   initializer=tf.constant(norm_values))

        with tf.name_scope('weightnorm') as scope:
            norms = tf.sqrt(tf.reduce_sum(tf.square(weight), 
                                          reduction_indices=[0]))
            weight = weight * (target_norms / norms)

    # if 'Discriminator' in name:
    #     print "WARNING weight constraint on {}".format(name)
    #     weight = tf.nn.softsign(10.*weight)*.1

    if inputs.get_shape().ndims == 2:
        result = tf.matmul(inputs, weight)
    else:
        reshaped_inputs = tf.reshape(inputs, [-1, input_dim])
        result = tf.matmul(reshaped_inputs, weight)
        result = tf.reshape(result, tf.pack(tf.unpack(tf.shape(inputs))[:-1] + [output_dim]))

    if biases:
        bias = tf.get_variable(name + '.b',
                               shape=output_dim,
                               initializer=tf.zeros_initializer())
        result = tf.nn.bias_add(result, bias)

    return result

def conv(x, ksize, stride, f_in, f_out, padding='SAME', use_bias=True,
         seed=None, name='conv'):

    shape = [ksize, ksize, f_in, f_out]
    initializer = tf.contrib.layers.variance_scaling_initializer(seed=seed, mode="FAN_AVG")
    weights = tf.get_variable(name + '_W',
                            shape=shape,
                            dtype='float',
                            initializer=initializer)
    x = tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding=padding)
    tf.summary.histogram(name+"_weights", weights)
    if use_bias:
        bias = tf.get_variable(name + '_b',
                               shape=[f_out],
                               dtype='float',
                               initializer=tf.constant_initializer(0.0))
        return tf.nn.bias_add(x, bias)
    else:
        return x
              
def conv_v2(x, ksize, stride, f_in, f_out, rng, he_init=True, bias=True, 
            name='conv', weightnorm=None):
    """
    x: tensor of shape (batch size, num channels, height, width)

    returns: tensor of shape (batch size, num channels, height, width)
    """

    def uniform(stdev, size):
        return rng.uniform(
            low=-stdev * np.sqrt(3),
            high=stdev * np.sqrt(3),
            size=size
        ).astype('float32')

    fan_in = f_in * ksize**2
    fan_out = f_out * ksize**2 / (stride**2)

    if he_init:
        filters_stdev = np.sqrt(4./(fan_in+fan_out))
    else: # Normalized init (Glorot & Bengio)
        filters_stdev = np.sqrt(2./(fan_in+fan_out))

    if _weights_stdev is not None:
        filter_values = uniform(
            _weights_stdev,
            (ksize, ksize, f_in, f_out)
        )
    else:
        filter_values = uniform(
            filters_stdev,
            (ksize, ksize, f_in, f_out)
        )

    filters = tf.get_variable(name + '.W',
                            initializer=tf.constant(filter_values))

    if weightnorm==None:
        weightnorm = _default_weightnorm
    if weightnorm:
        norm_values = np.sqrt(np.sum(np.square(filter_values), axis=(0,1,2)))
        target_norms = tf.get_variable(name + '.g',
                                       initializer=tf.constant(norm_values))
        with tf.name_scope('weightnorm') as scope:
            norms = tf.sqrt(tf.reduce_sum(tf.square(filters),
                                          reduction_indices=[0,1,2]))
            filters = filters * (target_norms / norms)

    result = tf.nn.conv2d(
        input=x, 
        filter=filters, 
        strides=[1, stride, stride, 1],
        padding='SAME',
        data_format='NHWC'
    )
    if bias:
        biases = tf.get_variable(name + '_b',
                                 shape=[f_out],
                                 initializer=tf.constant_initializer(0.0))
        result = tf.nn.bias_add(result, biases, data_format='NHWC')

    return result

def deconv(x, ksize, stride, f_in, f_out, use_bias=True, 
           seed=None, name='deconv'):

    shape = [ksize, ksize, f_out, f_in]
    initializer = tf.contrib.layers.variance_scaling_initializer(seed=seed, mode="FAN_AVG")
    weights = tf.get_variable(name + '_W',
                              shape=shape,
                              dtype='float',
                              initializer=initializer)
    input_shape = tf.shape(x)
    output_shape = tf.stack([input_shape[0], 2*input_shape[1], 
                             2*input_shape[2], f_out])
    
    x = tf.nn.conv2d_transpose(
        value=x, 
        filter=weights,
        output_shape=output_shape, 
        strides=[1, 2, 2, 1],
        padding='SAME'
    )

    if use_bias:
        bias = tf.get_variable(name + '_b',
                               dtype=tf.float32,
                               shape=[f_out],
                               initializer=tf.constant_initializer(0.0))
        return tf.nn.bias_add(x, bias)
    else:
        return x


def avg_pool(x, ksize=2, stride=2, name="avgpool"):
    return tf.nn.avg_pool(x,
                          ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1],
                          padding='SAME', name=name)


def max_pool(x, ksize=2, stride=2, name="pool"):
    return tf.nn.max_pool(x,
                          ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1],
                          padding='SAME', name=name)


def ce_loss(logit, y):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=y))


def accuracy(logit, y):
    pred = tf.argmax(logit, 1)
    true = tf.argmax(y, 1)
    return tf.reduce_mean(tf.to_float(tf.equal(pred, true)))

  
def logsoftmax(x):
    xdev = x - tf.reduce_max(x, 1, keep_dims=True)
    lsm = xdev - tf.log(tf.reduce_sum(tf.exp(xdev), 1, keep_dims=True))
    return lsm
  

def kl_divergence_with_logit(q_logit, p_logit):
    q = tf.nn.softmax(q_logit)
    qlogq = tf.reduce_mean(tf.reduce_sum(q * logsoftmax(q_logit), 1))
    qlogp = tf.reduce_mean(tf.reduce_sum(q * logsoftmax(p_logit), 1))
    return qlogq - qlogp


def entropy_y_x(logit):
    p = tf.nn.softmax(logit)
    return -tf.reduce_mean(tf.reduce_sum(p * logsoftmax(logit), 1))