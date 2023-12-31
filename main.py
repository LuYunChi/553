import os

import models
import utils
import numpy as np
import tensorflow as tf

from tqdm import tqdm

class FLAGS:
  dataset = 'cifar10-lt'
  data_home = 'data'
  train_batch_size = 128
  test_batch_size = 100
  mode = 'loss'  # ['baseline', 'posthoc', 'loss']
  tau = 1.0
  tb_log_dir = 'log'


def main():

  # Prepare the datasets.
  dataset = utils.dataset_mappings()[FLAGS.dataset]
  batches_per_epoch = int(dataset.num_train / FLAGS.train_batch_size)
  train_dataset = utils.create_tf_dataset(dataset, FLAGS.data_home,
                                          FLAGS.train_batch_size, True)
  test_dataset = utils.create_tf_dataset(dataset, FLAGS.data_home,
                                         FLAGS.test_batch_size, False)

  # Model to be trained.
  model = models.cifar_resnet32(dataset.num_classes)

  # Read the base probabilities to use for logit adjustment.
  base_probs_path = os.path.join(FLAGS.data_home,
                                 f'{FLAGS.dataset}_base_probs.txt')
  try:
    with tf.io.gfile.GFile(base_probs_path, mode='r') as fin:
      base_probs = np.loadtxt(fin)
  except tf.errors.NotFoundError:
    if FLAGS.mode in ['posthoc', 'loss']:
      raise
    else:
      base_probs = None

  # Build the loss function.
  loss_fn = build_loss_fn(FLAGS.mode == 'loss', base_probs, FLAGS.tau)

  # Prepare the metrics, the optimizer, etc.
  train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
  test_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
  posthoc_adjusting = FLAGS.mode == 'posthoc'
  if posthoc_adjusting:
    test_adj_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

  learning_rate = utils.LearningRateSchedule(
      schedule=dataset.lr_schedule,
      steps_per_epoch=batches_per_epoch,
      base_learning_rate=0.1,
  )

  optimizer = tf.keras.optimizers.SGD(
      learning_rate,
      momentum=0.9,
      nesterov=True,
  )

  # Prepare Tensorboard summary writers.
  train_summary_writer = tf.summary.create_file_writer(
      os.path.join(FLAGS.tb_log_dir, 'train'))
  test_summary_writer = tf.summary.create_file_writer(
      os.path.join(FLAGS.tb_log_dir, 'test'))

  # Train for num_epochs iterations over the train set.
  for epoch in tqdm(range(dataset.num_epochs)):

    # Iterate over the train dataset.
    for step, (x, y) in enumerate(train_dataset):
      with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)
        loss_value = loss_value + tf.reduce_sum(model.losses)

      grads = tape.gradient(loss_value, model.trainable_weights)
      optimizer.apply_gradients(zip(grads, model.trainable_weights))

      train_acc_metric.update_state(y, logits)

      # Log every 1000 batches.
      if step % 1000 == 0:
        print(f'Training loss (for one batch) at step {step}: {loss_value:.4f}')
        with train_summary_writer.as_default():
          tf.summary.scalar(
              'batch loss', loss_value, step=epoch * batches_per_epoch + step)

    # Display train metrics at the end of each epoch.
    train_acc = train_acc_metric.result()
    train_acc_metric.reset_states()
    print(f'Training accuracy over epoch: {train_acc:.4f}')
    with train_summary_writer.as_default():
      tf.summary.scalar(
          'accuracy', train_acc, step=(epoch + 1) * batches_per_epoch)

    # Run a test loop at the end of each epoch.
    for x, y in test_dataset:
      logits = model(x, training=False)
      test_acc_metric.update_state(y, logits)

      if posthoc_adjusting:
        # Posthoc logit-adjustment.
        adjusted_logits = logits - tf.math.log(
            tf.cast(base_probs**FLAGS.tau + 1e-12, dtype=tf.float32))
        test_adj_acc_metric.update_state(y, adjusted_logits)

    # Display test metrics.
    test_acc = test_acc_metric.result()
    test_acc_metric.reset_states()
    print(f'Test accuracy: {test_acc:.4f}')
    with test_summary_writer.as_default():
      tf.summary.scalar(
          'accuracy', test_acc, step=(epoch + 1) * batches_per_epoch)

    if posthoc_adjusting:
      test_adj_acc = test_adj_acc_metric.result()
      test_adj_acc_metric.reset_states()
      print(f'Logit-adjusted test accuracy: {test_adj_acc:.4f}')

      with test_summary_writer.as_default():
        tf.summary.scalar(
            'logit-adjusted accuracy',
            test_adj_acc,
            step=(epoch + 1) * batches_per_epoch)


def build_loss_fn(use_la_loss, base_probs, tau=1.0):
  """Builds the loss function to be used for training.

  Args:
    use_la_loss: Whether or not to use the logit-adjusted loss.
    base_probs: Base probabilities to use in the logit-adjusted loss.
    tau: Temperature scaling parameter for the base probabilities.

  Returns:
    A loss function with signature loss(labels, logits).
  """
  def loss_fn(labels, logits):
    if use_la_loss:
      logits = logits + tf.math.log(
          tf.cast(base_probs**tau + 1e-12, dtype=tf.float32))
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits)
    return tf.reduce_mean(loss, axis=0)

  return loss_fn

if __name__ == '__main__':
  main()
