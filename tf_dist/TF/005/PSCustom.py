# Based on: https://www.tensorflow.org/tutorials/distribute/parameter_server_training
import portpicker
import multiprocessing
import os
import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
import random

def create_in_process_cluster(num_workers, num_ps):
  """Creates and starts local servers and returns the cluster_resolver."""
  worker_ports = [portpicker.pick_unused_port() for _ in range(num_workers)]
  ps_ports = [portpicker.pick_unused_port() for _ in range(num_ps)]

  cluster_dict = {}
  cluster_dict["worker"] = ["localhost:%s" % port for port in worker_ports]
  if num_ps > 0:
    cluster_dict["ps"] = ["localhost:%s" % port for port in ps_ports]

  cluster_spec = tf.train.ClusterSpec(cluster_dict)

  # Workers need some inter_ops threads to work properly.
  worker_config = tf.compat.v1.ConfigProto()
  if multiprocessing.cpu_count() < num_workers + 1:
    worker_config.inter_op_parallelism_threads = num_workers + 1

  for i in range(num_workers):
    tf.distribute.Server(
        cluster_spec,
        job_name="worker",
        task_index=i,
        config=worker_config,
        protocol="grpc")

  for i in range(num_ps):
    tf.distribute.Server(
        cluster_spec,
        job_name="ps",
        task_index=i,
        protocol="grpc")

  cluster_resolver = tf.distribute.cluster_resolver.SimpleClusterResolver(
      cluster_spec, rpc_layer="grpc")
  return cluster_resolver

# Set the environment variable to allow reporting worker and ps failure to the
# coordinator. This is a workaround and won't be necessary in the future.
os.environ["GRPC_FAIL_FAST"] = "use_caller"

NUM_WORKERS = 3
NUM_PS = 1
cluster_resolver = create_in_process_cluster(NUM_WORKERS, NUM_PS)

variable_partitioner = (
    tf.distribute.experimental.partitioners.MinSizePartitioner(
        min_shard_bytes=(256 << 10),
        max_shards=NUM_PS))

strategy = tf.distribute.ParameterServerStrategy(
    cluster_resolver,
    variable_partitioner=variable_partitioner)


feature_vocab = [
    "avenger", "ironman", "batman", "hulk", "spiderman", "kingkong", "wonder_woman"
]
label_vocab = ["yes", "no"]

with strategy.scope():
  feature_lookup_layer = tf.keras.layers.StringLookup(
      vocabulary=feature_vocab,
      mask_token=None)
  label_lookup_layer = tf.keras.layers.StringLookup(
      vocabulary=label_vocab,
      num_oov_indices=0,
      mask_token=None)

  raw_feature_input = tf.keras.layers.Input(
      shape=(3,),
      dtype=tf.string,
      name="feature")
  feature_id_input = feature_lookup_layer(raw_feature_input)
  feature_preprocess_stage = tf.keras.Model(
      {"features": raw_feature_input},
      feature_id_input)

  raw_label_input = tf.keras.layers.Input(
      shape=(1,),
      dtype=tf.string,
      name="label")
  label_id_input = label_lookup_layer(raw_label_input)

  label_preprocess_stage = tf.keras.Model(
      {"label": raw_label_input},
      label_id_input)

def feature_and_label_gen(num_examples=200):
  examples = {"features": [], "label": []}
  for _ in range(num_examples):
    features = random.sample(feature_vocab, 3)
    label = ["yes"] if "avenger" in features else ["no"]
    examples["features"].append(features)
    examples["label"].append(label)
  return examples

examples = feature_and_label_gen()

def dataset_fn(_):
  raw_dataset = tf.data.Dataset.from_tensor_slices(examples)

  train_dataset = raw_dataset.map(
      lambda x: (
          {"features": feature_preprocess_stage(x["features"])},
          label_preprocess_stage(x["label"])
      )).shuffle(200).batch(32).repeat()
  return train_dataset

# These variables created under the `Strategy.scope` will be placed on parameter
# servers in a round-robin fashion.
with strategy.scope():
  # Create the model. The input needs to be compatible with Keras processing layers.
  model_input = tf.keras.layers.Input(
      shape=(3,), dtype=tf.int64, name="model_input")

  emb_layer = tf.keras.layers.Embedding(
      input_dim=len(feature_lookup_layer.get_vocabulary()), output_dim=16384)
  emb_output = tf.reduce_mean(emb_layer(model_input), axis=1)
  dense_output = tf.keras.layers.Dense(
      units=1, activation="sigmoid",
      kernel_regularizer=tf.keras.regularizers.L2(1e-4),
  )(emb_output)
  model = tf.keras.Model({"features": model_input}, dense_output)

  optimizer = tf.keras.optimizers.legacy.RMSprop(learning_rate=0.1)
  accuracy = tf.keras.metrics.Accuracy()

# These variables created under the `Strategy.scope` will be placed on parameter
# servers in a round-robin fashion.
with strategy.scope():
  # Create the model. The input needs to be compatible with Keras processing layers.
  model_input = tf.keras.layers.Input(
      shape=(3,), dtype=tf.int64, name="model_input")

  emb_layer = tf.keras.layers.Embedding(
      input_dim=len(feature_lookup_layer.get_vocabulary()), output_dim=16384)
  emb_output = tf.reduce_mean(emb_layer(model_input), axis=1)
  dense_output = tf.keras.layers.Dense(
      units=1, activation="sigmoid",
      kernel_regularizer=tf.keras.regularizers.L2(1e-4),
  )(emb_output)
  model = tf.keras.Model({"features": model_input}, dense_output)

  optimizer = tf.keras.optimizers.legacy.RMSprop(learning_rate=0.1)
  accuracy = tf.keras.metrics.Accuracy()
  

#assert len(emb_layer.weights) == 2
#assert emb_layer.weights[0].shape == (4, 16384)
#assert emb_layer.weights[1].shape == (4, 16384)

#print(emb_layer.weights[0].device)
#print(emb_layer.weights[1].device)

@tf.function
def step_fn(iterator):

  def replica_fn(batch_data, labels):
    with tf.GradientTape() as tape:
      pred = model(batch_data, training=True)
      per_example_loss = tf.keras.losses.BinaryCrossentropy(
          reduction=tf.keras.losses.Reduction.NONE)(labels, pred)
      loss = tf.nn.compute_average_loss(per_example_loss)
      model_losses = model.losses
      if model_losses:
        loss += tf.nn.scale_regularization_loss(tf.add_n(model_losses))
    gradients = tape.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    actual_pred = tf.cast(tf.greater(pred, 0.5), tf.int64)
    accuracy.update_state(labels, actual_pred)
    return loss

  batch_data, labels = next(iterator)
  losses = strategy.run(replica_fn, args=(batch_data, labels))
  return strategy.reduce(tf.distribute.ReduceOp.SUM, losses, axis=None)

coordinator = tf.distribute.coordinator.ClusterCoordinator(strategy)

@tf.function
def per_worker_dataset_fn():
  return strategy.distribute_datasets_from_function(dataset_fn)

per_worker_dataset = coordinator.create_per_worker_dataset(per_worker_dataset_fn)
per_worker_iterator = iter(per_worker_dataset)

num_epochs = 4
steps_per_epoch = 5
for i in range(num_epochs):
  accuracy.reset_states()
  for _ in range(steps_per_epoch):
    coordinator.schedule(step_fn, args=(per_worker_iterator,))
  # Wait at epoch boundaries.
  coordinator.join()
  print("Finished epoch %d, accuracy is %f." % (i, accuracy.result().numpy()))

loss = coordinator.schedule(step_fn, args=(per_worker_iterator,))
print("Final loss is %f" % loss.fetch())
