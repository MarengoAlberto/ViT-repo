import json
import os
import tensorflow.compat.v1 as tf
tf_config_str = os.environ.get('TF_CONFIG')
tf_config_dict  = json.loads(tf_config_str)
#print(json.dumps(tf_config_dict, indent=2))
tpu_config_str = os.environ.get('TPU_CONFIG')
tpu_config_dict  = json.loads(tpu_config_str)
#print(json.dumps(tpu_config_dict, indent=2))
tpu_name = tpu_config_dict["tpu_node_name"]
project_name = tpu_config_dict["project"]
zone_name = tpu_config_dict["zone"]
tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu_name, zone=zone_name, project=project_name)
#print(tpu_cluster_resolver.cluster_spec())
worker_list=tpu_cluster_resolver.cluster_spec()
#print(vars(worker_list))
print(worker_list._cluster_spec["worker"][0])
