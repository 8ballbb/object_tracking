import numpy as np
import cv2


class Detection:
    """
    This class represents a bounding box detection in a single image.

    Parameters
    ----------
    bbox : array_like
        Bounding box in format `(x, y, w, h)`.
    conf : float
        Detector confidence score.
    feature : array_like
        A feature vector that describes the object contained in this image.

    Attributes
    ----------
    bbox : ndarray
        Bounding box in format `(top left x, top left y, width, height)`.
    conf : ndarray
        Detector confidence score.
    class_name : ndarray
        Detector class.
    feature : ndarray | NoneType
        A feature vector that describes the object contained in this image.

    """

    def __init__(self, bbox, conf, class_name, feature):
        self.bbox = np.asarray(bbox, dtype=np.float)
        self.conf = float(conf)
        self.class_name = class_name
        self.feature = np.asarray(feature, dtype=np.float32)

    def get_class(self):
        return self.class_name

    def to_tlbr(self):
        """
        Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.bbox.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        """
        Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = self.bbox.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret


def non_max_suppression(detections, max_bbox_overlap=1.0):
    """Suppress overlapping detections.

    Parameters
    ----------
    detections : array
        Array of Detection objects
    max_bbox_overlap : float
        ROIs that overlap more than this values are suppressed

    Returns
    -------
    List[int]
        Returns array of Detection objects that have survived non-maxima suppression.
    """
    if len(detections) == 0:
        return []
    else:
        bboxes, scores = [], []
        for d in detections:
            bboxes.append(d.bbox)
            scores.append(d.conf)
        bboxes = np.array(bboxes).astype(np.float)
        indices = []

        x1 = bboxes[:, 0]
        y1 = bboxes[:, 1]
        x2 = bboxes[:, 2] + bboxes[:, 0]
        y2 = bboxes[:, 3] + bboxes[:, 1]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(scores)
        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            indices.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(
                idxs, np.concatenate(
                    ([last], np.where(overlap > max_bbox_overlap)[0])))
        detections = [detections[i] for i in indices]
        return detections
