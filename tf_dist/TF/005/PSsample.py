# This example creates a slurm cluster according to Slurm's environment
# variables and user input. A variable resides on a parameter server and workers
# repeatedly increments the value.
import os
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

tf.app.flags.DEFINE_integer('num_gpus', 1, 'How many GPUs to use.')
tf.app.flags.DEFINE_string('protocol', 'grpc', 'Communication protocol.')
tf.app.flags.DEFINE_integer('iters', 500, 'Number of iterations.')
FLAGS = tf.app.flags.FLAGS

def build_graph(task_index):
  with tf.variable_scope('ps'):
    with tf.device('/job:ps/task:0/cpu:0'):
      counter = tf.get_variable(name='counter',
                                shape=[],
                                initializer=tf.zeros_initializer(),
                                dtype=tf.float32,
                                use_resource=True)

  with tf.variable_scope('worker-'+str(task_index)):
    with tf.device('/job:worker/task:%d/gpu:0' % task_index):
      increment = tf.get_local_variable(name='increment',
                                        shape=[],
                                        initializer=tf.ones_initializer(),
                                        dtype=tf.float32,
                                        use_resource=True)
      add_op = tf.assign_add(counter, increment, use_locking=True)

  return add_op

def create_cluster():
  #  Jobs in a cluster resolver are specified in form of a dictionary
  #  where the keys are job names and values are number of tasks in each job.
  #  If task_per_node is not set it is automatically extract from Slurm's output
  #  environment variable. The number of tasks per node must be set such that
  #  it is divisible by the number of GPUs per node. auto_set_gpu is True by
  #  default which means that GPUs on nodes will be automatically assigned to
  #  tasks by setting CUDA_VISIBLE_DEVICE. In this example one GPU is allocated
  #  to each task. The method task_info returns two values: a string that
  #  represents the job name and the task index of the task of the calling process.

  cluster_resolver = tf.distribute.cluster_resolver.SlurmClusterResolver(
      {'ps': 1, 'worker': int(os.environ['SLURM_NTASKS'])-1},
      port_base=8888,
      tasks_per_node=2,
      gpus_per_node=2,
      gpus_per_task=1,
      auto_set_gpu=True)

  #Cluster is created when cluster_spec() is called, get_task_info()
  #should be called after calling cluster_spec()

  cluster = cluster_resolver.cluster_spec()
  job_name, task_index = cluster_resolver.get_task_info()

  config = tf.ConfigProto(allow_soft_placement=False,
                          log_device_placement=False)
  return config, cluster, job_name, task_index


def run_ps(task_index, cluster, config):
  server = tf.train.Server(cluster.as_cluster_def(),
                           job_name='ps',
                           task_index=task_index,
                           config=config,
                           protocol=FLAGS.protocol)
  server.join()

def run_worker(task_index, cluster, config):
  g = tf.get_default_graph()
  with g.as_default():
    add_op = build_graph(task_index)

  server = tf.train.Server(cluster.as_cluster_def(),
                           job_name='worker',
                           task_index=task_index,
                           config=config,
                           protocol=FLAGS.protocol)

  with tf.train.MonitoredTrainingSession(master=server.target,
                                         is_chief=(task_index == 0),
                                         config=config) as sess:
    for i in range(FLAGS.iters):
      val = sess.run(add_op)
      print('worker=%d, iter=%d, val=%f' % (task_index, i, val))

def main(argv):
  tf.logging.set_verbosity(tf.logging.INFO)
  config, cluster, job_name, task_index = create_cluster()

  if job_name == 'ps':
    run_ps(task_index, cluster, config)
  else:
    run_worker(task_index, cluster, config)

if __name__ == '__main__':
  tf.app.run(main=main, argv=None)
