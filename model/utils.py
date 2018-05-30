import tensorflow as tf
import math


def retrieve_seq_length_op3(data, pad_val=0):
    data_shape_size = data.get_shape().ndims
    if data_shape_size == 3:
        return tf.reduce_sum(tf.cast(tf.reduce_any(tf.not_equal(data, pad_val), axis=2), dtype=tf.int32), 1)
    elif data_shape_size == 2:
        return tf.reduce_sum(tf.cast(tf.not_equal(data, pad_val), dtype=tf.int32), 1)
    elif data_shape_size == 1:
        raise ValueError("retrieve_seq_length_op3: data has wrong shape!")
    else:
        raise ValueError(
            "retrieve_seq_length_op3: handling data_shape_size %s hasn't been implemented!" % (data_shape_size))


def multiply_tensors(tensor1, tensor2):
    """Multiplies two tensors in a matrix-like multiplication based on the
       last dimension of the first tensor and first dimension of the second
       tensor.

       Inputs:
            tensor1: A tensor of shape [a, b, c, .., x]
            tensor2: A tensor of shape [x, d, e, f, ...]

       Outputs:
            A tensor of shape [a, b, c, ..., d, e, f, ...]
    """
    sh1 = tf.shape(tensor1)
    sh2 = tf.shape(tensor2)
    len_sh1 = len(tensor1.get_shape())
    len_sh2 = len(tensor2.get_shape())
    prod1 = tf.constant(1, dtype=tf.int32)
    sh1_list = []
    for z in range(len_sh1 - 1):
        sh1_z = sh1[z]
        prod1 *= sh1_z
        sh1_list.append(sh1_z)
    prod2 = tf.constant(1, dtype=tf.int32)
    sh2_list = []
    for z in range(len_sh2 - 1):
        sh2_z = sh2[len_sh2 - 1 - z]
        prod2 *= sh2_z
        sh2_list.append(sh2_z)
    reshape_1 = tf.reshape(tensor1, [prod1, sh1[len_sh1 - 1]])
    reshape_2 = tf.reshape(tensor2, [sh2[0], prod2])
    result = tf.reshape(tf.matmul(reshape_1, reshape_2), sh1_list + sh2_list)
    assert len(result.get_shape()) == len_sh1 + len_sh2 - 2
    return result


def optimize_loss_batch_norm(loss, opt_name, cur_lr, max_grad_norm=None, move_avg_decay=None, momentum=0.9,
                             opt_decay=0.9,
                             global_step_val=0):
    global_step = tf.Variable(global_step_val, name="global_step", trainable=False)
    if opt_name == 'adam':
        optimizer = tf.train.AdamOptimizer(cur_lr)
    elif opt_name == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(cur_lr)
    elif opt_name == 'momentum':
        optimizer = tf.train.MomentumOptimizer(cur_lr, momentum, use_nesterov=True)
    elif opt_name == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(cur_lr)
    elif opt_name == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(cur_lr)
    elif opt_name == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(cur_lr, decay=opt_decay, momentum=momentum)
    else:
        raise ValueError("invalid optimizer %s" % opt_name)
    print("build optimizer ", optimizer)
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        if max_grad_norm is not None and max_grad_norm > 0:
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), max_grad_norm)
            train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)
            print("apply grad norm ", max_grad_norm)
        else:
            grads_and_vars = optimizer.compute_gradients(loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        if move_avg_decay is not None and move_avg_decay > 0:
            variable_averages = tf.train.ExponentialMovingAverage(move_avg_decay, global_step)
            variables_averages_op = variable_averages.apply(tf.trainable_variables())
            train_op = tf.group(train_op, variables_averages_op)
            print("apply move avg decay ", move_avg_decay)
    return train_op, global_step


def mask_inf(matrix, mask):
    negmask = 1 - mask
    num = 3.4 * math.pow(10, 38)
    return (matrix * mask) + (-((negmask * num + num) - num))