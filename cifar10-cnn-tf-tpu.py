from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import sys
import os
import time
import argparse
import math

import tensorflow as tf
from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator

import resnet_model

FLAGS = None
batch_size = 128 # this value is changed later
num_epochs = 10

monitor_loss = None
monitor_step = None

def _my_input_fn(params):
    del params # unused, but needed for TPU training

    image_bytes = 32 * 32 * 3
    label_bytes = 1
    
    def parser(serialized_example):
        train_bytes = tf.decode_raw(serialized_example, tf.uint8)
        train_label_uint8 = tf.strided_slice(
            train_bytes,
            [0],
            [label_bytes])
        train_image_uint8 = tf.strided_slice(
            train_bytes,
            [label_bytes],
            [label_bytes + image_bytes])
        train_label = tf.cast(train_label_uint8, tf.int32)
        train_label.set_shape([1])
        train_image_pre1 = tf.reshape(
            train_image_uint8,
            [3, 32, 32])
        # [depth, height, width] -> [height, width, depth]
        train_image_pre2 = tf.transpose(
            train_image_pre1,
            [1, 2, 0])
        train_image_pre3 = tf.cast(
            train_image_pre2,
            tf.float32)
        train_image = tf.image.per_image_standardization(train_image_pre3)
        train_image.set_shape([32, 32, 3])
        # convert label : (ex) 2 -> (0.0, 0.0, 1.0, 0.0, ...)
        train_label = tf.sparse_to_dense(train_label, [10], 1.0, 0.0)
        return train_image, train_label

    filenames = [os.path.join(FLAGS.train_dir, 'data_batch_%d.bin' % i)
        for i in range(1, 6)]
    dataset = tf.data.FixedLengthRecordDataset(
        filenames,
        image_bytes + label_bytes)
    dataset = dataset.prefetch(buffer_size=batch_size)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.map(
        parser,
        num_parallel_calls=batch_size)
    dataset = dataset.prefetch(4 * batch_size).cache().repeat()
    dataset = dataset.apply(
        tf.contrib.data.batch_and_drop_remainder(batch_size)
    )
    dataset = dataset.prefetch(1)
    return dataset

def _my_model_fn(features, labels, mode, params):
    del params # unused, but needed for TPU training

    #
    # Model - Here we use pre-built 'resnet_model'
    #
    model_params = resnet_model_tpu.HParams(
        batch_size=batch_size / FLAGS.num_replica, # because batch is divided by TPU replicas
        num_classes=10,
        min_lrn_rate=0.0001,
        lrn_rate=0.1,
        num_residual_units=5, # 5 x (3 x sub 2) + 2 = 32 layers
        use_bottleneck=False,
        weight_decay_rate=0.0002,
        relu_leakiness=0.1,
        optimizer='mom')
    train_model = resnet_model_tpu.ResNet(
        model_params,
        features,
        labels,
        'train')
    train_model.build_graph(tpu_opt=True)

    # create evaluation metrices
    truth = tf.argmax(train_model.labels, axis=1)
    predictions = tf.argmax(train_model.predictions, axis=1)
    #precision = tf.reduce_mean(
    #    tf.to_float(tf.equal(predictions, truth)),
    #    name="precision")
    #accuracy = tf.metrics.accuracy(truth, predictions)
    #tf.summary.scalar('accuracy', accuracy[1]) # output to TensorBoard

    # define operations (Here we assume only training operation !)
    #prediction_outputs = {
    #    "precision": precision
    #}
    return tpu_estimator.TPUEstimatorSpec(
        mode=mode,
        loss=train_model.cost,
        train_op=train_model.train_op,
        #predictions=prediction_outputs,
        eval_metrics=(
            metric_fn,
            [train_model.labels, train_model.predictions]))

def metric_fn(labels, logits):
    predictions = tf.argmax(logits, 1)
    return {
        'accuracy': tf.metrics.precision(
            labels=labels, predictions=predictions)
    }

def main(_):
    # define
    tpu_grpc_url = tf.contrib.cluster_resolver.TPUClusterResolver(
        tpu=["demo-tpu"]).get_master()
    model_dir = os.path.join(FLAGS.out_dir, str(int(time.time()))) + "/"
    run_config = tpu_config.RunConfig(
        master=tpu_grpc_url,
        model_dir=model_dir,
        save_checkpoints_secs=3600,
        session_config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True),
        tpu_config=tpu_config.TPUConfig(
            iterations_per_loop=100,
            num_shards=FLAGS.num_replica),
    )
    cifar10_resnet_classifier = tpu_estimator.TPUEstimator(
        model_fn=_my_model_fn,
        use_tpu=True,
        config=run_config,
        train_batch_size=batch_size)
        
    # run !
    cifar10_resnet_classifier.train(
        input_fn=_my_input_fn,
        #max_steps=50000 * 10 // batch_size) # Full spec
        max_steps=5000) # For benchmarking
        #max_steps=1000) # For seminar

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_dir',
        type=str,
        default='gs://cloud-tpu-demo/',
        help='Dir path for the training data.')
    parser.add_argument(
        '--out_dir',
        type=str,
        default='gs://cloud-tpu-demo/out/',
        help='Dir path for model output.')
    parser.add_argument(
        '--num_replica',
        type=int,
        default=1,
        help='Number of TPU chips replica')
    FLAGS, unparsed = parser.parse_known_args()

    """ batch_size must be the multiple of replica size """
    batch_size = batch_size * FLAGS.num_replica
    
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
