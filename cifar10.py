from six.moves import xrange
import os
import better_exceptions
import tensorflow as tf
import numpy as np
from tqdm import tqdm

from model import VQVAE, _cifar10_arch

# The codes are borrowed from
# https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10.py
# https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_input.py
DATA_DIR = 'datasets/cifar10'
DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
def maybe_download_and_extract():
    import sys, tarfile
    from six.moves import urllib
    """Download and extract the tarball from Alex's website."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    extracted_dir_path = os.path.join(DATA_DIR, 'cifar-10-batches-bin')
    if not os.path.exists(extracted_dir_path):
        tarfile.open(filepath, 'r:gz').extractall(DATA_DIR)

def read_cifar10(filename_queue):
    class CIFAR10Record(object):
        pass
    result = CIFAR10Record()
    record_bytes = 1 + 32*32*3

    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)
    record_bytes = tf.decode_raw(value, tf.uint8)

    result.label = tf.cast(
        tf.strided_slice(record_bytes, [0], [1]), tf.int32)
    depth_major = tf.reshape(
        tf.strided_slice(record_bytes, [1],
                         [1 + 32*32*3]),
        [3, 32, 32])
    # Convert from [depth, height, width] to [height, width, depth].
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])
    return result

def get_image(train=True,num_epochs=None):
    maybe_download_and_extract()
    if train:
        filenames = [os.path.join(DATA_DIR, 'cifar-10-batches-bin', 'data_batch_%d.bin' % i) for i in xrange(1, 6)]
    else:
        filenames = [os.path.join(DATA_DIR, 'cifar-10-batches-bin', 'test_batch.bin')]
    filename_queue = tf.train.string_input_producer(filenames,num_epochs=num_epochs)
    read_input = read_cifar10(filename_queue)
    return tf.cast(read_input.uint8image, tf.float32) / 255.0


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
    image = get_image()
    images = tf.train.shuffle_batch(
        [image],
        batch_size=BATCH_SIZE,
        num_threads=4,
        capacity=BATCH_SIZE*10,
        min_after_dequeue=BATCH_SIZE*2)
    valid_image = get_image(False)
    valid_images = tf.train.shuffle_batch(
        [valid_image],
        batch_size=BATCH_SIZE,
        num_threads=1,
        capacity=BATCH_SIZE*10,
        min_after_dequeue=BATCH_SIZE*2)
    # <<<<<<<

    # >>>>>>> MODEL
    with tf.variable_scope('train'):
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(LEARNING_RATE, global_step, DECAY_STEPS, DECAY_VAL, staircase=DECAY_STAIRCASE)
        tf.summary.scalar('lr',learning_rate)

        with tf.variable_scope('params') as params:
            pass
        net = VQVAE(learning_rate,global_step,BETA,images,K,D,_cifar10_arch,params,True)

    with tf.variable_scope('valid'):
        params.reuse_variables()
        valid_net = VQVAE(None,None,BETA,valid_images,K,D,_cifar10_arch,params,False)

    with tf.variable_scope('misc'):
        # Summary Operations
        tf.summary.scalar('loss',net.loss)
        tf.summary.scalar('recon',net.recon)
        tf.summary.scalar('vq',net.vq)
        tf.summary.scalar('commit',BETA*net.commit)
        tf.summary.scalar('nll',tf.reduce_mean(net.nll))
        tf.summary.image('origin',images,max_outputs=4)
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
            tf.summary.scalar('valid_nll',tf.reduce_mean(valid_net.nll)),
            tf.summary.image('valid_origin',valid_images,max_outputs=4),
            tf.summary.image('valid_recon',valid_net.p_x_z,max_outputs=4),
        ])

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Run!
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.graph.finalize()
    sess.run(init_op)

    summary_writer = tf.summary.FileWriter(LOG_DIR,sess.graph)
    summary_writer.add_summary(config_summary.eval(session=sess))

    try:
        # Start Queueing
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord,sess=sess)
        for step in tqdm(xrange(TRAIN_NUM),dynamic_ncols=True):
            it,loss,_ = sess.run([global_step,net.loss,net.train_op])

            if( it % SAVE_PERIOD == 0 ):
                net.save(sess,LOG_DIR,step=it)

            if( it % SUMMARY_PERIOD == 0 ):
                tqdm.write('[%5d] Loss: %1.3f'%(it,loss))
                summary = sess.run(summary_op)
                summary_writer.add_summary(summary,it)

            if( it % (SUMMARY_PERIOD*2) == 0 ): #Extended Summary
                summary = sess.run(extended_summary_op)
                summary_writer.add_summary(summary,it)

    except Exception as e:
        coord.request_stop(e)
    finally :
        net.save(sess,LOG_DIR)

        coord.request_stop()
        coord.join(threads)

    net.save(sess,LOG_DIR)

def test(MODEL,
         BETA,
         K,
         D,
         **kwargs):
    # >>>>>>> DATASET
    image = get_image(num_epochs=1)
    images = tf.train.batch(
        [image],
        batch_size=100,
        num_threads=1,
        capacity=100,
        allow_smaller_final_batch=True)
    valid_image = get_image(False,num_epochs=1)
    valid_images = tf.train.batch(
        [valid_image],
        batch_size=100,
        num_threads=1,
        capacity=100,
        allow_smaller_final_batch=True)
    # <<<<<<<

    # >>>>>>> MODEL
    with tf.variable_scope('net'):
        with tf.variable_scope('params') as params:
            pass
        x = tf.placeholder(tf.float32,[None,32,32,3])
        net= VQVAE(None,None,BETA,x,K,D,_cifar10_arch,params,False)

    init_op = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Run!
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.graph.finalize()
    sess.run(init_op)
    net.load(sess,MODEL)


    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord,sess=sess)
    try:
        nlls = []
        while not coord.should_stop():
            nlls.append(
                sess.run(net.nll,feed_dict={x:sess.run(valid_images)}))
            print('.', end='', flush=True)
    except tf.errors.OutOfRangeError:
        nlls = np.concatenate(nlls,axis=0)
        print(nlls.shape)
        print('NLL for test set: %f bits/dims'%(np.mean(nlls)))

    try:
        nlls = []
        while not coord.should_stop():
            nlls.append(
                sess.run(net.nll,feed_dict={x:sess.run(images)}))
            print('.', end='', flush=True)
    except tf.errors.OutOfRangeError:
        nlls = np.concatenate(nlls,axis=0)
        print(nlls.shape)
        print('NLL for training set: %f bits/dims'%(np.mean(nlls)))

    coord.request_stop()
    coord.join(threads)

def get_default_param():
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {
        'LOG_DIR':'./log/cifar10/%s'%(now),

        'TRAIN_NUM' : 250000, #Size corresponds to one epoch
        'BATCH_SIZE': 128,

        'LEARNING_RATE' : 0.0002,
        'DECAY_VAL' : 1.0,
        'DECAY_STEPS' : 20000, # Half of the training procedure.
        'DECAY_STAIRCASE' : False,

        'BETA':0.25,
        'K':10,
        'D':256,

        'SUMMARY_PERIOD' : 20,
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
    #test(MODEL='/home/nine/workspace/ndr/log/cifar10/2017-11-08 22:28:50/model.ckpt-150000',**config)
