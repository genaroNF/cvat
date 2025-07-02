
import io
import base64
import json
from itertools import chain

import cv2
import numpy as np
from ultralytics import YOLO

def init_context(context):
	model = YOLO('yolo11l-seg')
	context.user_data.model_handler = model

def handler(context, event):
	context.logger.info('Run YOLO11-seg model')
	data = event.body
	image_buffer = io.BytesIO(base64.b64decode(data['image']))
	image = cv2.imdecode(np.frombuffer(image_buffer.getvalue(), np.uint8), cv2.IMREAD_COLOR)

	results = context.user_data.model_handler.predict(image, conf=0.5)
	result = results[0]

	masks = []
	for mask in getattr(result.masks, "xy", []):
		masks.append(mask.tolist())

	confs = result.boxes.conf.tolist()
	clss = result.boxes.cls.tolist()
	class_name = result.names

	detections = []
	for mask, conf, cls in zip(masks, confs, clss):
		label = class_name[int(cls)]
		detections.append({
			'confidence': str(float(conf)),
			'label': label,
			'points': list(chain.from_iterable(mask)),
			'type': "polygon",
		})

	return context.Response(body=json.dumps(detections), headers={},
		content_type='application/json', status_code=200)