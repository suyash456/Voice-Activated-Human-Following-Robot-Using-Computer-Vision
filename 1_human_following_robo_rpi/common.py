# common.py

import numpy as np
from PIL import Image
from tensorflow.lite.python.interpreter import Interpreter
from tensorflow.lite.python.interpreter import load_delegate
import numpy as np
from PIL import Image

import os
import numpy as np
from PIL import Image


def load_model(model_dir, model_name, label_file, use_tpu=False):
    model_path = os.path.join(model_dir, model_name)
    label_path = os.path.join(model_dir, label_file)

    # Load model
    if use_tpu:
        interpreter = Interpreter(
            model_path=model_path,
            experimental_delegates=[load_delegate('libedgetpu.so.1')]
        )
    else:
        interpreter = Interpreter(model_path=model_path)

    interpreter.allocate_tensors()

    # Load labels
    with open(label_path, 'r') as f:
        lines = f.readlines()
        labels = {i: line.strip() for i, line in enumerate(lines)}

    return interpreter, labels


def load_labels(path='labels.txt'):
    with open(path, 'r') as f:
        lines = f.readlines()
        return {i: line.strip() for i, line in enumerate(lines)}

def set_input(interpreter, image):
    input_details = interpreter.get_input_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    # Resize the input image
    image = image.resize((width, height))

    input_tensor = np.expand_dims(np.array(image), axis=0)

    # Normalize if needed (depending on model)
    # input_tensor = (input_tensor / 255.0).astype(np.float32) # example

    interpreter.set_tensor(input_details[0]['index'], input_tensor)


def get_output(interpreter, score_threshold=0.5, top_k=10):
    output_details = interpreter.get_output_details()

    # Output tensors depend on the model, but typical SSD output:
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]       # Bounding boxes
    classes = interpreter.get_tensor(output_details[1]['index'])[0]     # Class indices
    scores = interpreter.get_tensor(output_details[2]['index'])[0]      # Scores
    count = int(interpreter.get_tensor(output_details[3]['index'])[0])  # Number of detections

    results = []
    for i in range(min(count, top_k)):
        if scores[i] >= score_threshold:
            result = {
                'bounding_box': boxes[i],
                'class_id': int(classes[i]),
                'score': scores[i]
            }
            results.append(result)

    return results
