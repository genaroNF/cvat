import io
import base64
import json
import torch

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results
from sahi.predict import get_sliced_prediction
from torch import tensor
from sahi import AutoDetectionModel

LABELS_DETECTION_MAP = {
    0: "Ant",
}


def sahi_result_to_ultralytics_results(image_np, sahi_results):
    return Results(
        image_np,
        "",
        LABELS_DETECTION_MAP,
        boxes=tensor(
            [
                [
                    bb.bbox.minx,
                    bb.bbox.miny,
                    bb.bbox.maxx,
                    bb.bbox.maxy,
                    bb.score.value,
                    bb.category.id,
                ]
                for bb in sahi_results.object_prediction_list
            ]
            if len(sahi_results.object_prediction_list) > 0
            else torch.empty((0, 6))
        ),
    )


# Initialize your model
def init_context(context):
    context.logger.info("Init context...  0%")
    model = YOLO("best.pt")
    context.user_data.model_handler = AutoDetectionModel.from_pretrained(
        model_type="yolov11", model=model
    )
    context.logger.info("Init context...100%")


# Inference endpoint
def handler(context, event):
    context.logger.info("Run custom yolov11 model")
    data = event.body
    image_buffer = io.BytesIO(base64.b64decode(data["image"]))
    image = cv2.imdecode(
        np.frombuffer(image_buffer.getvalue(), np.uint8), cv2.IMREAD_COLOR
    )

    sahi_results = get_sliced_prediction(
        image,
        context.user_data.model_handler,
        auto_slice_resolution=True,
        verbose=True,
    )

    result = sahi_result_to_ultralytics_results(image, sahi_results)

    boxes = []
    for box in getattr(result.boxes, "xyxy", []):
        boxes.append(box.tolist())

    confs = result.boxes.conf.tolist()
    clss = result.boxes.cls.tolist()
    class_name = result.names

    detections = []
    for box, conf, class_value in zip(boxes, confs, clss):
        label = class_name[int(class_value)]
        # must be in this format
        detections.append(
            {
                "confidence": str(float(conf)),
                "label": label,
                "points": box,
                "type": "rectangle",
            }
        )

    return context.Response(
        body=json.dumps(detections),
        headers={},
        content_type="application/json",
        status_code=200,
    )
