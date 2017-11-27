from six.moves import xrange
import os
import better_exceptions
import tensorflow as tf
import numpy as np
from tqdm import tqdm

from model import VQVAE, _imagenet_arch, PixelCNN

import sys
sys.path.append('slim_models/research/slim')
from datasets import imagenet
slim = tf.contrib.slim
def _build_batch(dataset,batch_size,num_threads):
    with tf.device('/cpu'):
        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            num_readers=num_threads,
            common_queue_capacity=20*batch_size,
            common_queue_min=10*batch_size,
            shuffle=True)
        image,label = provider.get(['image','label'])
        # Slim module has a background label as 0. By changing this, you need to use (label_num-1)
        # on Jupyter notebook to generate class conditioned samples.
        #label -= 1
        pp_image = tf.image.resize_images(image,[128,128]) / 255.0

        images,labels = tf.train.batch(
            [pp_image,label],
            batch_size=batch_size,
            num_threads=num_threads,
            capacity=5*batch_size,
            allow_smaller_final_batch=True)
        return images, labels

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
         SUMMARY_PERIOD,
         **kwargs):
    np.random.seed(RANDOM_SEED)
    tf.set_random_seed(RANDOM_SEED)

    # >>>>>>> DATASET
    train_dataset = imagenet.get_split('train','datasets/ILSVRC2012')
    valid_dataset = imagenet.get_split('validation','datasets/ILSVRC2012')
    train_ims,_ = _build_batch(train_dataset,BATCH_SIZE,4)
    valid_ims,_ = _build_batch(valid_dataset,4,1)

    # >>>>>>> MODEL
    with tf.variable_scope('train'):
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(LEARNING_RATE, global_step, DECAY_STEPS, DECAY_VAL, staircase=DECAY_STAIRCASE)
        tf.summary.scalar('lr',learning_rate)

        with tf.variable_scope('params') as params:
            pass
        net = VQVAE(learning_rate,global_step,BETA,train_ims,K,D,_imagenet_arch,params,True)

    with tf.variable_scope('valid'):
        params.reuse_variables()
        valid_net = VQVAE(None,None,BETA,valid_ims,K,D,_imagenet_arch,params,False)

    with tf.variable_scope('misc'):
        # Summary Operations
        tf.summary.scalar('loss',net.loss)
        tf.summary.scalar('recon',net.recon)
        tf.summary.scalar('vq',net.vq)
        tf.summary.scalar('commit',BETA*net.commit)
        tf.summary.scalar('nll',tf.reduce_mean(net.nll))
        tf.summary.image('origin',train_ims,max_outputs=4)
        tf.summary.image('recon',net.p_x_z,max_outputs=4)
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
            tf.summary.image('valid_origin',valid_ims,max_outputs=4),
            tf.summary.image('valid_recon',valid_net.p_x_z,max_outputs=4),
        ])
    # <<<<<<<<<<


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

def train_prior(config,
                RANDOM_SEED,
                MODEL,
                TRAIN_NUM,
                BATCH_SIZE,
                LEARNING_RATE,
                DECAY_VAL,
                DECAY_STEPS,
                DECAY_STAIRCASE,
                GRAD_CLIP,
                K,
                D,
                BETA,
                NUM_LAYERS,
                NUM_FEATURE_MAPS,
                SUMMARY_PERIOD,
                SAVE_PERIOD,
                **kwargs):
    np.random.seed(RANDOM_SEED)
    tf.set_random_seed(RANDOM_SEED)
    LOG_DIR = os.path.join(os.path.dirname(MODEL),'pixelcnn')
    # >>>>>>> DATASET
    train_dataset = imagenet.get_split('train','datasets/ILSVRC2012')
    ims,labels = _build_batch(train_dataset,BATCH_SIZE,4)
    # <<<<<<<

    # >>>>>>> MODEL for Generate Images
    with tf.variable_scope('net'):
        with tf.variable_scope('params') as params:
            pass
        vq_net = VQVAE(None,None,BETA,ims,K,D,_imagenet_arch,params,False)
    # <<<<<<<

    # >>>>>> MODEL for Training Prior
    with tf.variable_scope('pixelcnn'):
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(LEARNING_RATE, global_step, DECAY_STEPS, DECAY_VAL, staircase=DECAY_STAIRCASE)
        tf.summary.scalar('lr',learning_rate)

        net = PixelCNN(learning_rate,global_step,GRAD_CLIP,
                       vq_net.k.get_shape()[1],vq_net.embeds,K,D,
                       1000,NUM_LAYERS,NUM_FEATURE_MAPS)
    # <<<<<<
    with tf.variable_scope('misc'):
        # Summary Operations
        tf.summary.scalar('loss',net.loss)
        summary_op = tf.summary.merge_all()

        # Initialize op
        init_op = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())
        config_summary = tf.summary.text('TrainConfig', tf.convert_to_tensor(config.as_matrix()), collections=[])

        sample_images = tf.placeholder(tf.float32,[None,128,128,3])
        sample_summary_op = tf.summary.image('samples',sample_images,max_outputs=20)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Run!
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.graph.finalize()
    sess.run(init_op)
    vq_net.load(sess,MODEL)

    summary_writer = tf.summary.FileWriter(LOG_DIR,sess.graph)
    summary_writer.add_summary(config_summary.eval(session=sess))

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord,sess=sess)
    try:
        for step in tqdm(xrange(TRAIN_NUM),dynamic_ncols=True):
            batch_xs,batch_ys = sess.run([vq_net.k,labels])
            it,loss,_ = sess.run([global_step,net.loss,net.train_op],feed_dict={net.X:batch_xs,net.h:batch_ys})

            if( it % SAVE_PERIOD == 0 ):
                net.save(sess,LOG_DIR,step=it)
                sampled_zs,log_probs = net.sample_from_prior(sess,np.random.randint(0,1000,size=(10,)),2)
                sampled_ims = sess.run(vq_net.gen,feed_dict={vq_net.latent:sampled_zs})
                summary_writer.add_summary(
                    sess.run(sample_summary_op,feed_dict={sample_images:sampled_ims}),it)

            if( it % SUMMARY_PERIOD == 0 ):
                tqdm.write('[%5d] Loss: %1.3f'%(it,loss))
                summary = sess.run(summary_op,feed_dict={net.X:batch_xs,net.h:batch_ys})
                summary_writer.add_summary(summary,it)

    except Exception as e:
        coord.request_stop(e)
    finally :
        net.save(sess,LOG_DIR)

        coord.request_stop()
        coord.join(threads)

def get_default_param():
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {
        #'LOG_DIR':'./log/imagenet/%s'%('test'),
        'LOG_DIR':'./log/imagenet/%s'%(now),

        'TRAIN_NUM' : 50000, #Size corresponds to one epoch
        'BATCH_SIZE': 16,

        'LEARNING_RATE' : 0.0002,
        'DECAY_VAL' : 0.5,
        'DECAY_STEPS' : 25000, # Half of the training procedure.
        'DECAY_STAIRCASE' : False,

        'BETA':0.25,
        'K':512,
        'D':128,

        # PixelCNN Params
        'GRAD_CLIP' : 5.0,
        'NUM_LAYERS' : 18,
        'NUM_FEATURE_MAPS' : 256,

        'SUMMARY_PERIOD' : 50,
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
    config['LEARNING_RATE'] = 0.0004
    config['TRAIN_NUM'] = 300000
    config['BATCH_SIZE'] = 16
    config['DECAY_STEPS'] = 100000
    train_prior(config=config,**config)

    #TODO:
    # Reduce memory usage by batch learn batch_xs gathering process with batchsize 1
    # Only training for specific class labels. (1000 is too large classes)
    # Find correct ys...(Coral Reef, or something)

    #Warning:
    # Uncomment line 20 for training from scratch... The slim module assigns 0 for background.
