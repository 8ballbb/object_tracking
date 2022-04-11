import deepsort
import yolo
from time import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
# comment out below line to enable tensorflow logging outputs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import tensorflow.keras as keras
physical_devices = tf.config.experimental.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


def get_detections(
    frame, image_data, iou_threshold, score_threshold, framework, 
    input_size=None, input_details=None, output_details=None):
    if framework == "tflite":
        interpreter.set_tensor(input_details[0]["index"], image_data)
        interpreter.invoke()
        pred = [interpreter.get_tensor(output_details[i]["index"]) for i in range(len(output_details))]
        boxes, pred_conf = yolo.filter_boxes(
            pred[0], pred[1], 
            score_threshold=.25, 
            input_shape=tf.constant([input_size, input_size]))
    else:
        batch_data = tf.constant(image_data)
        pred_bbox = infer.predict(batch_data)
        for value in pred_bbox:
            temp_value = np.expand_dims(value, axis=0)
            boxes = temp_value[:, :, 0:4]
            pred_conf = temp_value[:, :, 4:]
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold
    )
    # convert data to numpy arrays and slice out unused elements
    num_objects = valid_detections.numpy()[0]
    bboxes = boxes.numpy()[0][:int(num_objects)]
    # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
    original_h, original_w, _ = frame.shape
    bboxes = yolo.format_boxes(bboxes, original_h, original_w)
    scores = scores.numpy()[0][:int(num_objects)]
    classes = classes.numpy()[0][:int(num_objects)]
    return bboxes, scores, classes, num_objects


def main(
    video, weights, input_size=416, iou_threshold=.45, score_threshold=.5, framework="tf", tiny=True):
    # Initiliase DeepSORT
    model_fname = "deepsort/model/mars-small128.pb"
    encoder = deepsort.box_encoder(model_fname, batch_size=1)
    metric = deepsort.CosineNearestNeighbor()  # calculate cosine distance metric
    tracker = deepsort.Tracker(metric)  # initialize tracker
    
    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    strides, anchors, num_class, xy_scale = yolo.load_config(tiny)
    # Load object detection model
    if framework == "tf":
        infer = keras.models.load_model(weights)
    elif framework == "tflite":
        interpreter = tf.lite.Interpreter(model_path=weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
    else:
        raise ValueError(f"framework `{framework}` not recognised")

    # video capture
    cap = cv2.VideoCapture(video)
    # video writer settings
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*"XVID")
    output = f"output/{video.split('/')[-1].split('.')[0]}.avi"
    print(f"Output file {output}")
    out = cv2.VideoWriter(output, codec, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            break
        # prepare frame for model
        image_data = cv2.resize(frame, (input_size, input_size)) / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time()
        # run detections
        if framework == "tflite": 
            bboxes, scores, classes, num_objects = get_detections(
                frame=frame, image_data=image_data, iou_threshold=iou_threshold, 
                score_threshold=score_threshold, framework=framework, 
                input_size=input_size, input_details=input_details, 
                output_details=output_details)
        else:
            bboxes, scores, classes, num_objects = get_detections(
                frame=frame, image_data=image_data, iou_threshold=iou_threshold, 
                score_threshold=score_threshold, framework=framework)
        # loop through objects and use class index to get class name
        class_names = yolo.read_class_names()
        names = np.array([class_names[int(classes[i])] for i in range(num_objects)])
        cv2.putText(
            frame, f"Objects being tracked: {len(names)}", 
            (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
        print("Objects being tracked: {}".format(len(names)))
        
        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [
            deepsort.Detection(*pred_info)
            for pred_info in zip(bboxes, scores, names, features)]
        # non-maxima supression
        detections = deepsort.non_max_suppression(detections)
        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        # initialize color map
        cmap = plt.get_cmap("tab20b")
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()

            # draw bbox on screen
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(
                frame, 
                (int(bbox[0]), int(bbox[1])), 
                (int(bbox[2]), int(bbox[3])), 
                color, .5)
            cv2.rectangle(
                frame, 
                (int(bbox[0]), int(bbox[1]-30)), 
                (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), 
                color, -1)
            cv2.putText(
                frame, 
                class_name + "-" + str(track.track_id),
                (int(bbox[0]), int(bbox[1]-10)),
                0, 0.75, (255, 255, 255), .5)
        # calculate frames per second of running detections
        fps = 1.0 / (time() - start_time)
        print("FPS: %.2f" % fps)
        out.write(cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR))
