import os
import time
import cv2
import numpy as np
from PIL import Image
from threading import Thread

from tensorflow.lite.python.interpreter import Interpreter
from tensorflow.lite.python.interpreter import load_delegate

#__________________Raspi Config__________________#

import RPi.GPIO as GPIO
import time
 
#GPIO Mode (BOARD / BCM)
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

 # Pins for Motor Driver Inputs 
Motor1A = 26
Motor1B = 19
Motor2A = 13
Motor2B = 6

GPIO.setup(Motor1A,GPIO.OUT)  # All pins as Outputs
GPIO.setup(Motor1B,GPIO.OUT)
GPIO.setup(Motor2A,GPIO.OUT)  # All pins as Outputs
GPIO.setup(Motor2B,GPIO.OUT)



# ---- Configuration ----
MODEL_DIR    = 'models/'
MODEL_FILE   = 'mobilenet_ssd_v2_coco_quant_postprocess.tflite'
LABEL_FILE   = 'coco_labels.txt'
THRESHOLD    = 0.2
TOP_K        = 10
TOLERANCE    = 0.1
VALID_CLASSES = ['person']

# ---- Motor / tracking state ----
x_deviation = 0.0
y_deviation = 0.0
arr_track_data = [0, 0, 0, 0, None, 0]

# ---- Helper functions ----

def load_model(model_dir, model_name, label_file, use_tpu=False):
    model_path = os.path.join(model_dir, model_name)
    label_path = os.path.join(model_dir, label_file)

    if use_tpu:
        interpreter = Interpreter(
            model_path=model_path,
            experimental_delegates=[load_delegate('libedgetpu.so.1')]
        )
    else:
        interpreter = Interpreter(model_path=model_path)

    interpreter.allocate_tensors()

    labels = {}
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                idx, name = parts
                labels[int(idx)] = name

    return interpreter, labels


def set_input(interpreter, image: Image.Image):
    input_details = interpreter.get_input_details()[0]
    height, width = input_details['shape'][1], input_details['shape'][2]

    img_resized = image.resize((width, height))
    input_tensor = np.expand_dims(np.array(img_resized), axis=0)

    interpreter.set_tensor(input_details['index'], input_tensor)


def get_output(interpreter, score_threshold=THRESHOLD, top_k=TOP_K):
    output_details = interpreter.get_output_details()
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]
    count = int(interpreter.get_tensor(output_details[3]['index'])[0])

    results = []
    for i in range(min(count, top_k)):
        if scores[i] >= score_threshold:
            results.append({
                'bounding_box': boxes[i].tolist(),
                'class_id': int(classes[i]),
                'score': float(scores[i])
            })
    return results


def get_delay(deviation, direction):
    d = 0.03
    dev = abs(deviation)
    if direction in ('f', 'b'):
        if dev >= 0.3: d = 0.1
        elif dev >= 0.2: d = 0.075
        elif dev >= 0.15: d = 0.045
        else: d = 0.035
    else:
        if dev >= 0.4: d = 0.08
        elif dev >= 0.35: d = 0.07
        elif dev >= 0.3: d = 0.06
        elif dev >= 0.25: d = 0.05
        elif dev >= 0.2: d = 0.04
    return d


def move_robot():
    global x_deviation, y_deviation, arr_track_data
    cmd = None
    delay = 0

    if abs(x_deviation) < TOLERANCE and abs(y_deviation) < TOLERANCE:
        cmd = 'Stop'
        GPIO.output(Motor1A,GPIO.LOW)
        GPIO.output(Motor1B,GPIO.LOW)
        GPIO.output(Motor2A,GPIO.LOW)
        GPIO.output(Motor2B,GPIO.LOW)
        
    else:
        if abs(x_deviation) > abs(y_deviation):
            if x_deviation > 0:
                cmd = 'Move Backward'
                GPIO.output(Motor1A,GPIO.HIGH)
                GPIO.output(Motor1B,GPIO.LOW)
                GPIO.output(Motor2A,GPIO.HIGH)
                GPIO.output(Motor2B,GPIO.LOW)
                delay = get_delay(x_deviation, 'l')
            else:
                cmd = 'Move Forward'
                GPIO.output(Motor1A,GPIO.LOW)
                GPIO.output(Motor1B,GPIO.HIGH)
                GPIO.output(Motor2A,GPIO.LOW)
                GPIO.output(Motor2B,GPIO.HIGH)
                delay = get_delay(x_deviation, 'r')
        else:
            if y_deviation > 0:
                cmd = 'Move Right'
                GPIO.output(Motor1A,GPIO.LOW)
                GPIO.output(Motor1B,GPIO.LOW)
                GPIO.output(Motor2A,GPIO.HIGH)
                GPIO.output(Motor2B,GPIO.LOW)
                delay = get_delay(y_deviation, 'f')
            else:
                cmd = 'Move Left'
                GPIO.output(Motor1A,GPIO.HIGH)
                GPIO.output(Motor1B,GPIO.LOW)
                GPIO.output(Motor2A,GPIO.LOW)
                GPIO.output(Motor2B,GPIO.LOW)
                delay = get_delay(y_deviation, 'b')

    # --- GPIO or Motor control logic goes here ---
    print(f"[Command]: {cmd}, [Delay]: {delay:.2f}s")
    # Example (for Raspberry Pi):
    # if cmd == 'Move Left':
    #     motor.turn_left()
    #     time.sleep(delay)
    # elif cmd == 'Move Right':
    #     motor.turn_right()
    # elif cmd == 'Move Forward':
    #     motor.forward()
    # elif cmd == 'Move Backward':
    #     motor.backward()
    # elif cmd == 'Stop':
    #     motor.stop()

    arr_track_data[4] = cmd
    arr_track_data[5] = delay


def track_object(objs, labels):
    global x_deviation, y_deviation, arr_track_data
    if not objs:
        print("No objects to track")
        arr_track_data = [0, 0, 0, 0, None, 0]
        return

    print("Detections:")
    for det in objs:
        cid = det['class_id']
        name = labels.get(cid, f"id_{cid}")
        print(f" - {name} (id={cid}) @ {det['score']:.2f}")

    target = None
    for det in objs:
        cid = det['class_id']
        name = labels.get(cid, "").lower()
        if name == 'person':
            target = det
            break

    if target is None:
        print("Selected object not present")
        return

    x0, y0, x1, y1 = target['bounding_box']
    x_center = (x0 + x1) / 2.0
    y_center = (y0 + y1) / 2.0
    x_deviation = round(0.5 - x_center, 3)
    y_deviation = round(0.5 - y_center, 3)

    print(f"Person @ center=({x_center:.3f}, {y_center:.3f}), dev=({x_deviation}, {y_deviation})")
    move_robot()

    arr_track_data[0:4] = [x_center, y_center, x_deviation, y_deviation]


def draw_overlays(frame, objs, labels, arr_dur, arr_track_data):
    h, w = frame.shape[:2]
    
    # Top overlay bar
    cv2.rectangle(frame, (0, 0), (w, 24), (0, 0, 0), -1)
    cam_ms = int(arr_dur[0] * 1000)
    inf_ms = int(arr_dur[1] * 1000)
    other_ms = int(arr_dur[2] * 1000)
    cv2.putText(frame, f"Cam:{cam_ms}ms Inf:{inf_ms}ms Oth:{other_ms}ms", (10, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Bottom overlay bar with deviation and direction
    cv2.rectangle(frame, (0, h - 48), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, f"X Dev: {arr_track_data[2]:.3f} | Y Dev: {arr_track_data[3]:.3f} | Tolerance: {TOLERANCE}",
                (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    direction_text = arr_track_data[4] or "No Movement"
    cv2.putText(frame, f"Direction: {direction_text}", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    for det in objs:
        x0, y0, x1, y1 = det['bounding_box']
        x0, y0 = int(x0 * w), int(y0 * h)
        x1, y1 = int(x1 * w), int(y1 * h)
        score = int(det['score'] * 100)
        name = labels.get(det['class_id'], '?')
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 150, 255), 2)
        cv2.putText(frame, f"{name} {score}%", (x0, y0 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return frame


def main():
    cap = cv2.VideoCapture(0)
    interpreter, labels = load_model(MODEL_DIR, MODEL_FILE, LABEL_FILE, use_tpu=False)
    arr_dur = [0, 0, 0]

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        # Flip horizontally (like a mirror)
        frame = cv2.flip(frame, 1)

        pil_im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        t1 = time.time()
        set_input(interpreter, pil_im)
        interpreter.invoke()
        objs = get_output(interpreter)
        t2 = time.time()

        track_object(objs, labels)
        t3 = time.time()

        arr_dur[0] = t1 - t0
        arr_dur[1] = t2 - t1
        arr_dur[2] = t3 - t2

        out = draw_overlays(frame, objs, labels, arr_dur, arr_track_data)
        cv2.imshow('Camera', out)

        print(f"FPS: {1.0 / (time.time() - t0):.1f}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()
