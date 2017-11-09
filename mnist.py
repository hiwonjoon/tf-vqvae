from six.moves import xrange
import better_exceptions
import tensorflow as tf
import numpy as np
from tqdm import tqdm

from model import VQVAE, _mnist_arch

def main(config,
         RANDOM_SEED,
         LOG_DIR,
         TRAIN_NUM,
         BATCH_SIZE,
         LEARNING_RATE,
         DECAY_VAL,
         DECAY_STEPS,
         DECAY_STAIRCASE,
         BETA,
         K,
         D,
         SAVE_PERIOD,
         SUMMARY_PERIOD):
    np.random.seed(RANDOM_SEED)
    tf.set_random_seed(RANDOM_SEED)

    # >>>>>>> DATASET
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("datasets/mnist", one_hot=True)
    # <<<<<<<

    # >>>>>>> MODEL
    x = tf.placeholder(tf.float32,[None,784])
    resized = tf.image.resize_images(
        tf.reshape(x,[-1,28,28,1]),
        (24,24),
        method=tf.image.ResizeMethod.BILINEAR)

    with tf.variable_scope('train'):
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(LEARNING_RATE, global_step, DECAY_STEPS, DECAY_VAL, staircase=DECAY_STAIRCASE)
        tf.summary.scalar('lr',learning_rate)

        with tf.variable_scope('params') as params:
            pass
        net = VQVAE(learning_rate,global_step,BETA,resized,K,D,_mnist_arch,params,True)

    with tf.variable_scope('valid'):
        params.reuse_variables()
        valid_net = VQVAE(None,None,BETA,resized,K,D,_mnist_arch,params,False)

    with tf.variable_scope('misc'):
        # Summary Operations
        tf.summary.scalar('loss',net.loss)
        tf.summary.scalar('recon',net.recon)
        tf.summary.scalar('vq',net.vq)
        tf.summary.scalar('commit',BETA*net.commit)
        tf.summary.image('origin',resized,max_outputs=4)
        tf.summary.image('recon',net.p_x_z,max_outputs=4)
        # TODO: logliklihood

        summary_op = tf.summary.merge_all()

        # Initialize op
        init_op = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())
        config_summary = tf.summary.text('TrainConfig', tf.convert_to_tensor(config.as_matrix()), collections=[])

        extended_summary_op = tf.summary.merge([
            tf.summary.scalar('valid_loss',valid_net.loss),
            tf.summary.scalar('valid_recon',valid_net.recon),
            tf.summary.scalar('valid_vq',valid_net.vq),
            tf.summary.scalar('valid_commit',BETA*valid_net.commit),
            tf.summary.image('valid_recon',valid_net.p_x_z,max_outputs=10),
        ])

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Run!
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.graph.finalize()
    sess.run(init_op)

    summary_writer = tf.summary.FileWriter(LOG_DIR,sess.graph)
    summary_writer.add_summary(config_summary.eval(session=sess))

    for step in tqdm(xrange(TRAIN_NUM),dynamic_ncols=True):
        batch_xs, _= mnist.train.next_batch(BATCH_SIZE)
        it,loss,_ = sess.run([global_step,net.loss,net.train_op],feed_dict={x:batch_xs})

        if( it % SAVE_PERIOD == 0 ):
            net.save(sess,LOG_DIR,step=it)

        if( it % SUMMARY_PERIOD == 0 ):
            tqdm.write('[%5d] Loss: %1.3f'%(it,loss))
            summary = sess.run(summary_op,feed_dict={x:batch_xs})
            summary_writer.add_summary(summary,it)

        if( it % (SUMMARY_PERIOD*2) == 0 ): #Extended Summary
            batch_xs, _= mnist.test.next_batch(BATCH_SIZE)
            summary = sess.run(extended_summary_op,feed_dict={x:batch_xs})
            summary_writer.add_summary(summary,it)

    net.save(sess,LOG_DIR)

def get_default_param():
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {
        'LOG_DIR':'./log/mnist/%s'%(now),

        'TRAIN_NUM' : 60000, #Size corresponds to one epoch
        'BATCH_SIZE': 32,

        'LEARNING_RATE' : 0.0002,
        'DECAY_VAL' : 1.0,
        'DECAY_STEPS' : 20000, # Half of the training procedure.
        'DECAY_STAIRCASE' : False,

        'BETA':0.25,
        'K':20,
        'D':64,

        'SUMMARY_PERIOD' : 100,
        'SAVE_PERIOD' : 10000,
        'RANDOM_SEED': 0,
    }

if __name__ == "__main__":
    class MyConfig(dict):
        pass
    params = get_default_param()
    config = MyConfig(params)
    def as_matrix() :
        return [[k, str(w)] for k, w in config.items()]
    config.as_matrix = as_matrix

    main(config=config,**config)
