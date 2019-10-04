import os

import tensorflow as tf
from tqdm import tqdm
from tflibs.runner import Runner
from tflibs.utils import distributed_run, unpack_dict
from tflibs.io import Reader
from tflibs.dataset import Split, DatasetSpec


class CheckpointHandler:
    def __init__(self, checkpoint: tf.train.Checkpoint, checkpoint_prefix: str):
        self._checkpoint = checkpoint
        self._checkpoint_prefix = checkpoint_prefix

    @property
    def checkpoint(self):
        return self._checkpoint

    @property
    def checkpoint_prefix(self):
        return self._checkpoint_prefix

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.checkpoint.save(file_prefix=self.checkpoint_prefix)
        print('Model saved when exiting')

        return False


def main(strategy: tf.distribute.MirroredStrategy,
         global_step: tf.Tensor,
         train_writer: tf.summary.SummaryWriter,
         eval_writer: tf.summary.SummaryWriter,
         train_batch_size: int,
         eval_batch_size: int,
         job_dir: str,
         dataset_dir: str,
         dataset_filename: str,
         num_epochs: int,
         summary_steps: int,
         log_steps: int,
         dataset_spec: DatasetSpec,
         model: tf.keras.Model,
         loss_fn: tf.keras.losses.Loss,
         optimizer: tf.keras.optimizers.Optimizer):
    # Define metrics
    eval_metric = tf.keras.metrics.CategoricalAccuracy()
    best_metric = tf.Variable(eval_metric.result())

    # Define training loop

    @distributed_run(strategy)
    def train_step(inputs):
        with tf.GradientTape() as tape:
            images, labels = inputs

            logits = model(images)

            cross_entropy = loss_fn(labels, logits)
            loss = tf.reduce_sum(cross_entropy) / train_batch_size

            gradients = tape.gradient(loss, model.variables)
            optimizer.apply_gradients(zip(gradients, model.variables))

            if global_step % summary_steps == 0:
                tf.summary.scalar('loss', loss, step=global_step)

            return loss

    @distributed_run(strategy)
    def eval_step(inputs, metric):
        images, labels = inputs

        logits = model(images)

        metric.update_state(labels, logits)

    # Build input pipeline
    train_reader = Reader(dataset_dir, dataset_filename, split=Split.Train)
    test_reader = Reader(dataset_dir, dataset_filename, split=Split.Test)
    train_dataset = train_reader.read()
    test_dataset = test_reader.read()

    @unpack_dict
    def map_fn(_id, image, label):
        return tf.cast(image, tf.float32) / 255., label

    train_dataset = dataset_spec.parse(train_dataset).batch(train_batch_size).map(map_fn)
    test_dataset = dataset_spec.parse(test_dataset).batch(eval_batch_size).map(map_fn)

    #################
    # Training loop #
    #################
    # Define checkpoint
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     model=model,
                                     global_step=global_step,
                                     best_metric=best_metric)
    # Restore the model
    checkpoint_dir = job_dir
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    # Prepare dataset for distributed run
    train_dataset = strategy.experimental_distribute_dataset(train_dataset)
    test_dataset = strategy.experimental_distribute_dataset(test_dataset)

    with CheckpointHandler(checkpoint, checkpoint_prefix):
        for epoch in range(num_epochs):
            print('---------- Epoch: {} ----------'.format(epoch + 1))

            print('Starting training for epoch: {}'.format(epoch + 1))
            with train_writer.as_default():
                for inputs in tqdm(train_dataset, initial=global_step.numpy(), desc='Training', unit=' steps'):
                    per_replica_losses = train_step(inputs)
                    mean_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, None)

                    if global_step.numpy() % log_steps == 0:
                        print('Loss: {}'.format(mean_loss.numpy()))

                    # Increment global step
                    global_step.assign_add(1)

            print('Starting evaluation for epoch: {}'.format(epoch + 1))

            with eval_writer.as_default():
                for inputs in tqdm(test_dataset, desc='Evaluating'):
                    eval_step(inputs, eval_metric)

                accuracy = eval_metric.result()
                print('Accuracy: {}'.format(accuracy.numpy()))
                tf.summary.scalar('accuracy', accuracy, step=global_step)

                if accuracy >= best_metric:
                    checkpoint.save(file_prefix=checkpoint_prefix + '-best')
                    print('The best model saved: {} is higher than {}'.format(accuracy.numpy(), best_metric.numpy()))
                    best_metric.assign(accuracy)

            eval_metric.reset_states()


if __name__ == '__main__':
    runner = Runner()

    runner.run(main)
