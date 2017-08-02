import tensorflow as tf
import fonts
import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from tensorflow.examples.tutorials.mnist import input_data
import layers as L
from tqdm import tqdm
import time
from matplotlib import colors
config_proto = tf.ConfigProto()
config_proto.gpu_options.allow_growth = True

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 32, "the number of examples in a batch")
tf.app.flags.DEFINE_integer('eval_freq', 250, "The number of iterations until evaluation")
tf.app.flags.DEFINE_integer('plot_freq', 250, "The number of iterations until evaluation")

tf.app.flags.DEFINE_integer('num_epochs', 30, "the number of epochs for training")
tf.app.flags.DEFINE_integer('dis_per_iter', 1, "the number of training cycles of the discriminator for each iteration")
tf.app.flags.DEFINE_integer('gen_per_iter', 1, "the number of training cycles of the generator for each iteration")

tf.app.flags.DEFINE_float('lr', 5e-5, "initial learning rate")
tf.app.flags.DEFINE_float('mom1', 0.9, "initial momentum rate")
tf.app.flags.DEFINE_float('mom2', 0.5, "momentum rate after epoch_decay_start")
tf.app.flags.DEFINE_boolean('dis_bn', True, "use batch norm at discriminator")
tf.app.flags.DEFINE_boolean('gen_bn', True, "use batch norm at generator")
tf.app.flags.DEFINE_boolean('method', {'gan', 'cgan', 'acgan', 'infogan'})
tf.app.flags.DEFINE_boolean('wasserstein', True, "use Wasserstein GAN method for improved convergence.")
tf.app.flags.DEFINE_boolean('gen_dropout', False, "use dropout in the generator.")
tf.app.flags.DEFINE_string('dataset', 'mnist', '{mnist, fonts}')
tf.app.flags.DEFINE_string('experiment_name', 'default', 'experiment name used for logging.')
tf.app.flags.DEFINE_string('log_dir', '/home/david/training_logs/GAN', 'experiment name used for logging.')
experiment_dir = FLAGS.log_dir + "/" + FLAGS.experiment_name
image_dir = experiment_dir + "/images/"
if not os.path.exists(image_dir):
    os.makedirs(image_dir)
if FLAGS.dataset == "mnist":
    mnist = input_data.read_data_sets('home/david/datasets/MNIST_data', one_hot=True, reshape=False,  dtype="uint8")
    Z_dim = 128
    X_dim = 28
    y_dim = 10
    h_dim = 128
    num_channels = 1
    iter_per_epoch = 50000 / FLAGS.batch_size
    images = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
    labels = tf.placeholder(tf.float32, shape=[None, y_dim])
elif FLAGS.dataset == "fonts":
    Z_dim = 128
    X_dim = 64
    y_dim = 62
    h_dim = 256
    num_channels = 1
    iter_per_epoch = 360000 / FLAGS.batch_size
    with tf.device("/cpu:0"):
        images, labels = fonts.inputs(one_hot=True, batch_size=FLAGS.batch_size, batch_ids=[0])
elif FLAGS.dataset == 'cifar10':
    import cifar10
    Z_dim = 128
    X_dim = 32
    y_dim = 10
    h_dim = 256
    num_channels = 3
    iter_per_epoch = 40000 / FLAGS.batch_size
    with tf.device("/cpu:0"):
        images, labels = cifar10.inputs(FLAGS.batch_size)
else:
    raise NotImplementedError
with open(os.path.join(experiment_dir, 'flags.txt'), 'w') as f:
    for key in vars(FLAGS)['__flags'].keys():
        print key, str(vars(FLAGS)['__flags'][key]) + "\n"
        f.write(str(key) + ", " + str(vars(FLAGS)['__flags'][key]) + " \n")
rng = np.random.RandomState(seed=1234)

def discriminator(x, y, is_training=True, update_batch_stats=True,
                  act_fn=L.lrelu, bn=FLAGS.dis_bn, reuse=True):
    with tf.variable_scope('discriminator', reuse=reuse):
        if FLAGS.method == 'cgan':
            h = L.fc(y, y_dim, X_dim*X_dim, seed=rng.randint(123456),
                 name='fc_y')
            h = tf.reshape(h, [-1, X_dim, X_dim, 1])
            h = tf.concat((x, h), axis=3)
            h = L.conv(h, 3, 1, num_channels+1, 32, name="conv1")
        else:
            h = L.conv(x, 3, 1, num_channels, 32, name="conv1")
        h = act_fn(h)

        # 64x64 -> 32x32
        h = L.conv(h, 4, 2, 32, 64, name="conv2", )
        h = L.bn(h, 64, is_training=is_training,
                     update_batch_stats=update_batch_stats,
                     use_gamma=False, name='bn1') if bn else h
        h = act_fn(h)

        # 32x32 -> 16x16
        h = L.conv(h, 4, 2, 64, 128, name="conv3")
        h = L.bn(h, 128, is_training=is_training, update_batch_stats=update_batch_stats,
                     use_gamma=False, name='bn2') if bn else h
        h = act_fn(h)
        h = L.conv(h, X_dim / 4, 1, 128, 1, name="conv5", padding="VALID")
        logits = tf.reshape(h, [-1, 1])
        return logits

def generator(z, y, is_training=True, update_batch_stats=True,
              act_fn=L.lrelu, bn=FLAGS.gen_bn,
              reuse=True, dropout=FLAGS.gen_dropout):
    with tf.variable_scope('generator', reuse=reuse):
        if FLAGS.method == "cgan":
            inputs = tf.concat(axis=1, values=[z, y])
            h = L.fc(inputs, Z_dim+y_dim, ((X_dim / 4)**2)*128, seed=rng.randint(123456), name='fc1')
        else:
            h = L.fc(z, Z_dim, ((X_dim / 4) ** 2) * 128, seed=rng.randint(123456), name='fc1')
        h = L.bn(h, ((X_dim / 4)**2)*128, is_training=is_training, update_batch_stats=update_batch_stats, use_gamma=False, name='bn1') if bn else h
        h = act_fn(h)
        h = tf.reshape(h, [-1, X_dim / 4, X_dim / 4, 128])

        # 16x16 -> 32x32
        h = L.deconv(h, ksize=2, stride=2, f_in=128, f_out=64, name="deconv1")
        h = L.conv(h, 5, 1, 64, 64, name="conv1")
        h = L.bn(h, 64, is_training=is_training, update_batch_stats=update_batch_stats,
                 use_gamma=False, name='bn2') if bn else h
        h = tf.nn.dropout(h, keep_prob=0.5) if dropout else h
        h = act_fn(h)


        h = L.conv(h, 3, 1, 64, 64, name="conv2")
        h = L.bn(h, 64, is_training=is_training, update_batch_stats=update_batch_stats, use_gamma=False, name='b3') if bn else h
        h = tf.nn.dropout(h, keep_prob=0.5) if dropout else h
        h = act_fn(h)

        # 32x32 -> 64x64
        h = L.deconv(h, ksize=2, stride=2, f_in=64, f_out=32, name="deconv2")
        h = L.conv(h, 5, 1, 32, 32, name="conv3")
        h = L.bn(h, 32, is_training=is_training, update_batch_stats=update_batch_stats, use_gamma=False, name='b4')
        h = tf.nn.dropout(h, keep_prob=0.5) if dropout else h
        h = act_fn(h)

        h = L.conv(h, 5, 1, 32, num_channels, name="conv4")
        h = tf.nn.tanh(h, name="output")
        return h
"""
def discriminator(x, y, act_fn=L.lrelu, reuse=True, is_training=True, update_batch_stats=True, bn=FLAGS.dis_bn):
    with tf.variable_scope('discriminator', reuse=reuse):
        x = tf.reshape(x, [-1, X_dim*X_dim])
        xy = tf.concat((x, y), axis=1)
        h = L.fc(xy, X_dim*X_dim+y_dim, 4*h_dim, name="fc1")
        h = act_fn(h)
        h = L.bn(h, 4*h_dim, is_training=is_training, update_batch_stats=update_batch_stats, name="bn1") if bn else h
        h = L.fc(h, 4*h_dim, 2*h_dim, name="fc2")
        h = act_fn(h)
        h = L.bn(h, 2*h_dim, is_training=is_training, update_batch_stats=update_batch_stats, name="bn2") if bn else h
        h = L.fc(h, 2*h_dim, h_dim, name="fc3")
        h = act_fn(h)
        h = L.bn(h, h_dim, is_training=is_training, update_batch_stats=update_batch_stats, name="bn3") if bn else h
        h = L.fc(h, h_dim, 1, name="fc4")
        return h

def generator(z, y, act_fn=L.lrelu, reuse=True, is_training=True, update_batch_stats=True, bn=FLAGS.gen_bn):
    with tf.variable_scope('generator', reuse=reuse):
        zy = tf.concat((z, y), axis=1)
        h = L.fc(zy, Z_dim + y_dim, h_dim, name="fc1")
        h = act_fn(h)
        h = L.bn(h, h_dim, is_training=is_training, update_batch_stats=update_batch_stats, name="bn1") if bn else h
        h = L.fc(h, h_dim, 2*h_dim, name="fc2")
        h = act_fn(h)
        h = L.bn(h, 2*h_dim, is_training=is_training, update_batch_stats=update_batch_stats, name="bn2") if bn else h
        h = L.fc(h, 2*h_dim, 4*h_dim, name="fc3")
        h = act_fn(h)
        h = L.bn(h, 4*h_dim, is_training=is_training, update_batch_stats=update_batch_stats, name="bn3") if bn else h
        h = L.fc(h, 4*h_dim, X_dim*X_dim, name="fc4")
        return tf.nn.tanh(h)

"""
def sample_Z(m, n):
    return tf.random_normal(shape=[m, n], mean=0., stddev=1.)

def plot(samples, iter=None, title=None):
    fig = plt.figure(figsize=(4, 4), facecolor='#edededff')

    if y_dim == 10:
        gs = gridspec.GridSpec(4, 3)
    else:
        gs = gridspec.GridSpec(8, 8)
    gs.update(wspace=0.05, hspace=0.05)
    if iter is not None:
        plt.suptitle("Iteration step: %s" % iter)

    for i, sample in enumerate(samples):
        img = (sample + 1.) * 0.5
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        if num_channels>1:
            plt.imshow(img.reshape(X_dim, X_dim, num_channels))
        else:
            plt.imshow(img.reshape(X_dim, X_dim), cmap='Greys_r')

    return fig


Z = sample_Z(FLAGS.batch_size, Z_dim)
Z_fixed = tf.constant(np.random.uniform(-1., 1., size=(y_dim, Z_dim)).astype("float32"))
labels_fixed = tf.one_hot(tf.cast(tf.range(0, y_dim), tf.int32), y_dim)

G_fixed = generator(Z_fixed, labels_fixed, reuse=None, bn=FLAGS.gen_bn, is_training=False, update_batch_stats=False)
G_sample = generator(Z, labels, bn=FLAGS.gen_bn)

logit_real = discriminator(images, labels, reuse=None)
logit_fake = discriminator(G_sample, labels, update_batch_stats=True)

# Gather parameters for gen/dis
gen_params = L.params_with_name('generator')
dis_params = L.params_with_name('discriminator')

logit_real_mean = tf.reduce_mean(logit_real)
logit_fake_mean = tf.reduce_mean(logit_fake)
if FLAGS.wasserstein:
    D_loss = - logit_real_mean + logit_fake_mean
    G_loss = - logit_fake_mean

    # theta_D is list of D's params
    clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in dis_params]

    D_solver = (tf.train.RMSPropOptimizer(learning_rate=FLAGS.lr)
                .minimize(D_loss, var_list=dis_params))
    G_solver = (tf.train.RMSPropOptimizer(learning_rate=FLAGS.lr)
                .minimize(G_loss, var_list=gen_params))
else:
    D_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=logit_real, labels=tf.ones([FLAGS.batch_size, 1])))
    D_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=logit_fake, labels=tf.zeros([FLAGS.batch_size, 1])))
    D_loss = D_loss_real + D_loss_fake
    G_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=logit_fake, labels=tf.ones([FLAGS.batch_size, 1])))
    D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=dis_params)
    G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=gen_params)

tf.summary.scalar("Avg_Logit_real", logit_real_mean)
tf.summary.scalar("Avg_Logit_fake", logit_fake_mean)

tf.summary.scalar("Disc_loss", D_loss)
tf.summary.scalar("Gen_loss", G_loss)

# add image summary
tf.summary.image('real', images)
tf.summary.image('fake', G_sample)

merged_summary_op = tf.summary.merge_all()
init_op = tf.global_variables_initializer()

# Create TF session and initialise variables
sess = tf.Session(config=config_proto)
sess.run(tf.global_variables_initializer())
summary_writer = tf.summary.FileWriter(experiment_dir, graph=tf.get_default_graph())
feed_dict = {}
real_images = []


with tf.device("/cpu:0"):
    tf.train.start_queue_runners(sess=sess)
with tf.device("/gpu:0"):
    l = 0
    while len(real_images) < y_dim:
        if FLAGS.dataset == "mnist":
            img, lab = mnist.train.next_batch(FLAGS.batch_size)
        else:
            img, lab = sess.run([images, labels])
        for i in range(img.shape[0]):
            label = np.argmax(lab[i, :]).astype("int32")
            if label == l:
                print "adding image for class %i" % l
                real_images.append(img[i])
                l += 1
    print "Creating real image for comparison"
    fig = plot(real_images)
    file_name = FLAGS.dataset + "-REAL_IMAGES.jpg"
    plt.savefig(image_dir + file_name, bbox_inches='tight')
    plt.close(fig)

    D_loss_sum = 0.
    G_loss_sum = 0.
    global_step = 0
    for ep in tqdm(range(FLAGS.num_epochs)):
        ep_time = time.time()
        for it in tqdm(range(iter_per_epoch)):

            if global_step % FLAGS.plot_freq == 0:
                samples = sess.run(G_fixed)
                fig = plot(samples, title="Generated samples for: "+FLAGS.experiment_name, iter=str(global_step).zfill(8))
                file_name = FLAGS.dataset + "-ep-"+str(ep).zfill(3)+"-iter-" + str(it).zfill(7) + ".png"
                plt.savefig(image_dir + file_name, bbox_inches='tight', transparent=False, facecolor='#edededff')
                plt.close(fig)

            for i in range(FLAGS.dis_per_iter):
                if FLAGS.dataset == "mnist":
                    x_mb, y_mb = mnist.train.next_batch(FLAGS.batch_size)
                    feed_dict = {images: (x_mb.astype("float32") / 127.5) - 1., labels: y_mb}
                if FLAGS.wasserstein:
                    _, D_loss_curr, _ = sess.run([D_solver, D_loss, clip_D], feed_dict=feed_dict)
                else:
                    _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict=feed_dict)
                D_loss_sum += D_loss_curr
            for i in range(FLAGS.gen_per_iter):
                _, G_loss_curr, summary = sess.run([G_solver, G_loss, merged_summary_op], feed_dict=feed_dict)
                G_loss_sum += G_loss_curr
            summary_writer.add_summary(summary, global_step)
            if it % FLAGS.eval_freq == 0:
                print '\nIter: {}'.format(it), 'D loss: {:.4}'.format(D_loss_sum/(FLAGS.dis_per_iter*FLAGS.eval_freq)),\
                    'G_loss: {:.4}'.format(G_loss_sum / (FLAGS.gen_per_iter*FLAGS.eval_freq)) + "\r"
                D_loss_sum = 0.
                G_loss_sum = 0.
            global_step += 1
        print "Epoch ", ep, "finished in :", time.time()-ep_time, "seconds"
