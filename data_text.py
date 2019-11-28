from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import pickle
import random
import glob 

import enum
import numpy as np
import six
import tensorflow as tf


NDIM = 640

ProblemInstance = collections.namedtuple(
    "ProblemInstance",
    ["tr_input", "tr_output", "tr_info", "val_input", "val_output", "val_info"])


class StrEnum(enum.Enum):
  """An Enum represented by a string."""

  def __str__(self):
    return self.value

  def __repr__(self):
    return self.__str__()


class EmbeddingCrop(StrEnum):
  """Embedding types supported by the DataProvider class."""
  CENTER = "center"
  MULTIVIEW = "multiview"


class MetaSplit(StrEnum):
  """Meta-datasets split supported by the DataProvider class."""
  TRAIN = "train"
  VALID = "val"
  TEST = "test"


class DataProvider(object):

  def __init__(self, dataset_split, config, verbose=False):

    self._dataset_split = MetaSplit(dataset_split)
    self._config = config
    self._verbose = verbose
    self._base_path = os.path.join(self._config['data_path'], self._config['dataset_name'])
    self._splitfilename = os.path.join(self._config['data_path'], 'reddit_splits/{}_{}.txt'.format(self._config['splitname'], self._dataset_split))
    self._class_dict = [c.strip().split(',') for c in open(self._splitfilename).readlines()]
    self._class_dict = {c[0]: c[1:] for c in self._class_dict}
    self.classes = list(self._class_dict.keys())
    self._ndim = self._config['ndim']
    self.multi_topic = self._config['multi_topic']


  def sample_emb(self, author_names, labels, nb_samples):

      samples = []

      for i, author_name in zip(labels, author_names):

          if self.multi_topic:
              topic = random.choice(self._class_dict[author_name])
          else:
              topic = self.classes_dict[author_name][0]

          emb = np.load(os.path.join(self.base_path, topic, '{}.np'.format(author_name)))
          idx = np.random.choice(emb.shape[0], nb_samples, replace = False)
          samples += [(i, e) for e in emb[idx]]

      if shuffle:
          random.shuffle(samples)

      return samples
  
  def get_instance(self, num_classes, tr_size, val_size):
    """Samples a random N-way K-shot classification problem instance.

    Args:
      num_classes: N in N-way classification.
      tr_size: K in K-shot; number of training examples per class.
      val_size: number of validation examples per class.

    Returns:
      A tuple with 6 Tensors with the following shapes:
      - tr_input: (num_classes, tr_size, NDIM): training image embeddings.
      - tr_output: (num_classes, tr_size, 1): training image labels.
      - tr_info: (num_classes, tr_size): training image file names.
      - val_input: (num_classes, val_size, NDIM): validation image embeddings.
      - val_output: (num_classes, val_size, 1): validation image labels.
      - val_input: (num_classes, val_size): validation image file names.
    """
    def _build_one_instance_py():
      
      paths = random.sample(self.classes, num_classes)
      idx = np.random.permutation(num_classes)
      labels = np.array(list(range(num_classes)))
      labels = labels[idx]
      emb_labels = self.sample_emb(paths, labels, tr_size + val_size):
      
      labels, emb = list(zip(*emb_labels))
      labels, emb= np.array(list(labels)), np.array(list(emb))
      labels = labels.reshape((num_classes, tr_size + val_size, 1))
      emb = emb.reshape((num_classes, tr_size + val_size, self._config['ndim']))

      return emb, labels, paths

    output_list = tf.py_func(_build_one_instance_py, [],
                             [tf.float32, tf.int32, tf.string])

    instance_input, instance_output, instance_info = output_list
    instance_input = tf.nn.l2_normalize(instance_input, axis=-1)

    split_sizes = [tr_size, val_size]
    tr_input, val_input = tf.split(instance_input, split_sizes, axis=1)
    tr_output, val_output = tf.split(instance_output, split_sizes, axis=1)
    
    with tf.control_dependencies(
        self._check_labels(num_classes, tr_size, val_size,
                           tr_output, val_output)):
      tr_output = tf.identity(tr_output)
      val_output = tf.identity(val_output)
    
    tr_info = instance_info
    val_info = instance_info


    return tr_input, tr_output, tr_info, val_input, val_output, val_info
  
  
  def _check_labels(self, num_classes, tr_size, val_size,
                    tr_output, val_output):
    correct_label_sum = (num_classes*(num_classes-1))//2
    tr_label_sum = tf.reduce_sum(tr_output)/tr_size
    val_label_sum = tf.reduce_sum(val_output)/val_size
    all_label_asserts = [
        tf.assert_equal(tf.to_int32(tr_label_sum), correct_label_sum),
        tf.assert_equal(tf.to_int32(val_label_sum), correct_label_sum),
    ]
    return all_label_asserts


  def get_batch(self, batch_size, num_classes, tr_size, val_size,
                num_threads=10):
    """Returns a batch of random N-way K-shot classification problem instances.

    Args:
      batch_size: number of problem instances in the batch.
      num_classes: N in N-way classification.
      tr_size: K in K-shot; number of training examples per class.
      val_size: number of validation examples per class.
      num_threads: number of threads used to sample problem instances in
      parallel.

    Returns:
      A ProblemInstance of Tensors with the following shapes:
      - tr_input: (batch_size, num_classes, tr_size, NDIM): training image
      embeddings.
      - tr_output: (batch_size, num_classes, tr_size, 1): training image
      labels.
      - tr_info: (batch_size, num_classes, tr_size): training image file
      names.
      - val_input: (batch_size, num_classes, val_size, NDIM): validation
      image embeddings.
      - val_output: (batch_size, num_classes, val_size, 1): validation
      image labels.
      - val_info: (batch_size, num_classes, val_size): validation image
      file names.
    """
    if self._verbose:
      num_threads = 1
    one_instance = self.get_instance(num_classes, tr_size, val_size)

    tr_data_size = (num_classes, tr_size)
    val_data_size = (num_classes, val_size)
    task_batch = tf.train.shuffle_batch(one_instance, batch_size=batch_size,
                                        capacity=1000, min_after_dequeue=0,
                                        enqueue_many=False,
                                        shapes=[tr_data_size + (self.ndim,),
                                                tr_data_size + (1,),
                                                (num_classes,),
                                                val_data_size + (self.ndim,),
                                                val_data_size + (1,),
                                                (num_classes,)],
                                        num_threads=num_threads)


    return ProblemInstance(*task_batch)

