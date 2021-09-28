import os
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
import cv2
import numpy as np
import tensorflow as tf


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

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-21')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

category_index = label_map_util.create_category_index_from_labelmap(ANNOTATION_PATH+'/label_map.pbtxt')


# Setup capture
cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

def overlay_transparent(background, overlay, x, y):

    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype=overlay.dtype) * 255
            ],
            axis = 2,
        )

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_image

    return background

def choose_emoji(classe):
    items = {1:"Peace",
            2:"Ok",
            3:"Hello",
            4:"Fist",
            5:"Heart",
            }

    emoji = items[classe+1]
    return cv2.imread(emoji+".png", -1)



while True:
    ret, frame = cap.read()
    #frame = cv2.resize(frame, (300, 300))
    frame = cv2.flip(frame, 1)
    image_np = np.array(frame)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    min_score_thresh = 0.3

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'] + label_id_offset,
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=2,
        min_score_thresh=min_score_thresh,
        agnostic_mode=False)

    boxes = detections['detection_boxes']
    scores = detections['detection_scores']
    classes = detections['detection_classes']

    im_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    im_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    final_box = []

    cpt_classe = 0
    for box in boxes[scores > min_score_thresh]:

        ymin, xmin, ymax, xmax = box
        xmin, xmax, ymin, ymax = int(xmin * im_width), int(xmax * im_width), int(ymin * im_height), int(ymax * im_height)

        #emoji = cv2.imread('Ok.png', -1)
        emoji = choose_emoji(classe=detections['detection_classes'][cpt_classe])
        emoji = cv2.resize(emoji, (150, 150))

        #image_np_with_detections = overlay_transparent(image_np_with_detections, img2, xmin+50, ymin+100)
        image_np_with_detections = overlay_transparent(image_np_with_detections, emoji, xmin-75+int(abs(xmax-xmin)/2), ymin-75+int(abs(ymax-ymin)/2))

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw the rectangle around each face

    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        emoji = cv2.imread("Head.png", -1)
        emoji = cv2.resize(emoji, (250, 250))
        image_np_with_detections = overlay_transparent(image_np_with_detections, emoji,
                                                       x - 125 + int(abs(w) / 2),
                                                       y - 125 + int(abs(h) / 2))


    cv2.imshow('object detection', image_np_with_detections)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        break