import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format


## Definition des paths ##

RACINE = "/Users/daoud.kadoch/PycharmProjects/HumanToEmoji/"
WORKSPACE_PATH = RACINE + "Tensorflow/workspace"
SCRIPTS_PATH = RACINE + "Tensorflow/scripts"
APIMODEL_PATH = RACINE + "Tensorflow/models"
ANNOTATION_PATH = WORKSPACE_PATH + '/annotations'
IMAGE_PATH = "/Users/daoud.kadoch/Documents/HumanToEmoji_dataset"
MODEL_PATH = WORKSPACE_PATH + '/models'
PRETRAINED_MODEL_PATH = WORKSPACE_PATH + '/pre-trained-models'
CONFIG_PATH = MODEL_PATH + '/mobilnet_ssd_human_to_emoji/pipeline.config'
CHECKPOINT_PATH = MODEL_PATH + '/mobilnet_ssd_human_to_emoji/'
CUSTOM_MODEL_NAME = 'mobilnet_ssd_human_to_emoji'


def main():
    ## LABELS MAP CREATION ##


    labels = [{'name': 'Peace', 'id': 1},
              {'name': 'Ok', 'id': 2},
              {'name': 'Hello', 'id': 3},
              {'name': 'Fist', 'id': 4},
              {'name': 'Heart', 'id': 5}]


    f = open(ANNOTATION_PATH + '/label_map.pbtxt', 'w')
    for label in labels:
        f.write('item { \n')
        f.write('\tname:\'{}\'\n'.format(label['name']))
        f.write('\tid:{}\n'.format(label['id']))
        f.write('}\n')

    ## TF RECORDS ##
    # Executer tf_records.sh

    CONFIG_PATH = MODEL_PATH+'/'+CUSTOM_MODEL_NAME+'/pipeline.config'
    config = config_util.get_configs_from_pipeline_file(CONFIG_PATH)

    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(CONFIG_PATH, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline_config)

    pipeline_config.model.ssd.num_classes = 5
    pipeline_config.train_config.batch_size = 4
    pipeline_config.train_config.fine_tune_checkpoint = PRETRAINED_MODEL_PATH + '/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/checkpoint/ckpt-0'
    pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
    pipeline_config.train_input_reader.label_map_path = ANNOTATION_PATH + '/label_map.pbtxt'
    pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [ANNOTATION_PATH + '/train.record']
    pipeline_config.eval_input_reader[0].label_map_path = ANNOTATION_PATH + '/label_map.pbtxt'
    pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [ANNOTATION_PATH + '/test.record']

    config_text = text_format.MessageToString(pipeline_config)
    with tf.io.gfile.GFile(CONFIG_PATH, "wb") as f:
        f.write(config_text)


    # Pour le training
    print("Commande à executer pour le training du modèle : \n")
    print("""python3 {}/research/object_detection/model_main_tf2.py --model_dir={}/{} --pipeline_config_path={}/{}/pipeline.config --num_train_steps=20000""".format(
            APIMODEL_PATH, MODEL_PATH, CUSTOM_MODEL_NAME, MODEL_PATH, CUSTOM_MODEL_NAME))

if __name__ == "__main__":
    main()
