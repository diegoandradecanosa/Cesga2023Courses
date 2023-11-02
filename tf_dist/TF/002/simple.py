import tensorflow as tf
import os
#tf.distribute.cluster_resolver.SlurmClusterResolver(jobs=None, port_base=8888, gpus_per_node=1, gpus_per_task=1,
#    tasks_per_node=1, auto_set_gpu=True, rpc_layer='grpc')
print(tf.distribute.cluster_resolver.SlurmClusterResolver().cluster_spec())
