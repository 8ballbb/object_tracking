import numpy as np


yolo_params = dict(
    classes_fname="",  # TODO: FILL IN
    strides=dict(
        tiny=[16, 32], normal=[8, 16, 32]
    ),
    anchors=dict(
        tiny=[23,27, 37,58, 81,82, 81,82, 135,169, 344,319], 
        normal=[12,16, 19,36, 40,28, 36,75, 76,55, 72,146, 142,110, 192,243, 459,401]),
    xy_scale=dict(
        tiny=[1.05, 1.05], 
        normal=[1.2, 1.1, 1.05])
)


def get_anchors(model_type):
    anchors_path = yolo_params["anchors"][model_type]
    anchors = np.array(anchors_path)
    return anchors.reshape(2, 3, 2) if model_type == "tiny" else anchors.reshape(3, 3, 2)


def read_class_names():
    """TODO: rewrite function"""
    names = {}
    with open(yolo_params["classes_fname"], "r") as f:
        for ID, name in enumerate(f):
            names[ID] = name.strip("\n")
    return names


def load_config(tiny):
    model_type = "tiny" if tiny else "normal"
    strides = np.array(yolo_params["strides"][model_type])
    anchors = get_anchors(model_type)
    xy_scale = yolo_params["xy_scale"][model_type]
    num_class = len(read_class_names())
    return strides, anchors, num_class, xy_scale


def format_boxes(bboxes, image_height, image_width):
    for box in bboxes:
        ymin = int(box[0] * image_height)
        xmin = int(box[1] * image_width)
        ymax = int(box[2] * image_height)
        xmax = int(box[3] * image_width)
        width = xmax - xmin
        height = ymax - ymin
        box[0], box[1], box[2], box[3] = xmin, ymin, width, height
    return bboxes
