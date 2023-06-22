import argparse
import gstreamer
import os
import time
import json
import operator
import paho.mqtt.client as mqtt
import struct

from common import avg_fps_counter, SVG
from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference

TOPIC = "becherlager"
TOPIC_INT = "cupholder"
TOPIC_COUNT = "cupholder_count"
BROKER_ADRESS = "172.19.12.128"
PORT = 1883
QOS = 1
CAM_W, CAM_H = 640, 480
DEFAULT_MODEL_DIR = "models"
DEFAULT_MODEL = "cinito_vision_edgetpu.tflite"
DEFAULT_LABELS = "cinito_labels.txt"
FILE_PATH = "/home/mendel/cinito-vision/resources/cup_positions.json"

def generate_svg(src_size, inference_box, objs, labels, text_lines):
    svg = SVG(src_size)
    src_w, src_h = src_size
    box_x, box_y, box_w, box_h = inference_box
    scale_x, scale_y = src_w / box_w, src_h / box_h

    for y, line in enumerate(text_lines, start=1):
        svg.add_text(10, y * 20, line, 20)
    for obj in objs:
        bbox = obj.bbox
        if not bbox.valid:
            continue
        # Absolute coordinates, input tensor space.
        x, y = bbox.xmin, bbox.ymin
        w, h = bbox.width, bbox.height
        # Subtract boxing offset.
        x, y = x - box_x, y - box_y
        # Scale to source coordinate space.
        x, y, w, h = x * scale_x, y * scale_y, w * scale_x, h * scale_y
        percent = int(100 * obj.score)
        label = "{}% {}".format(percent, labels.get(obj.id, obj.id))
        svg.add_text(x, y - 5, label, 20)
        svg.add_rect(x, y, w, h, "red", 2)
    return svg.finish()


def centerIsInside(cup, basket):
    """
    Cup is the list with x1, y1, x2 and y2 for that cup
    Basket is the list with x1, y1, x2 and y2 for that position in the basket
    """
    # Center of the cup:
    c_xcenter = (cup[0] + cup[2]) / 2
    c_ycenter = (cup[1] + cup[3]) / 2

    positions = {}
    for i, basket_pos in enumerate(basket):
        h_xmin = min(basket_pos[0], basket_pos[2])
        h_xmax = max(basket_pos[0], basket_pos[2])
        h_ymin = min(basket_pos[1], basket_pos[3])
        h_ymax = max(basket_pos[1], basket_pos[3])

        in_range_along_x = c_xcenter < h_xmax and h_xmin < c_xcenter
        in_range_along_y = c_ycenter < h_ymax and h_ymin < c_ycenter

        if in_range_along_x and in_range_along_y:
            positions[("id_" + str(i))] = True
            return i

    return -1


def sorted_bbox(cup_bbox):
    cup_bbox = sorted(cup_bbox, key=operator.itemgetter(1))
    for i in range(4):
        start = i * 4
        end = i * 4 + 4
        row = cup_bbox[start:end]
        row = sorted(row, key=operator.itemgetter(0))
        cup_bbox[start:end] = row

    return cup_bbox


def check_cup_bbox(cup_bbox, basket):
    cup_bbox = sorted_bbox(cup_bbox)
    for cup in cup_bbox:
        positions = centerIsInside(cup, basket)

    return positions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        help=".tflite model path",
        default=os.path.join(DEFAULT_MODEL_DIR, DEFAULT_MODEL),
    )
    parser.add_argument(
        "--labels",
        help="label file path",
        default=os.path.join(DEFAULT_MODEL_DIR, DEFAULT_LABELS),
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="number of categories with highest score to display",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.60, help="classifier score threshold"
    )
    parser.add_argument(
        "--videosrc", help="Which video source to use. ", default="/dev/video0"
    )
    parser.add_argument(
        "--videofmt",
        help="Input video format.",
        default="raw",
        choices=["raw", "h264", "jpeg"],
    )
    parser.add_argument(
        "--init",
        type=bool,
        default=False,
        help="initialises the reference positions of the individual cups",
    )
    args = parser.parse_args()
    
    print("Loading {} with {} labels.".format(args.model, args.labels))
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    labels = read_label_file(args.labels)
    inference_size = input_size(interpreter)

    fps_counter = avg_fps_counter(30)

    cup_bbox, args.init = get_reference_positions(args)

    # Create a MQTT client
    client = mqtt.Client()

    # Set up the callback functions
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect

    client.connect(BROKER_ADRESS, PORT)
    # client.loop_start()
    # detect_cups(args, client)
    # client.loop_stop()

    def user_callback(input_tensor, src_size, inference_box):
        client.publish(TOPIC_INT, 43, qos=QOS)
        
        start_time = time.monotonic()
        run_inference(interpreter, input_tensor)
        # For larger input image sizes, use the edgetpu.classification.engine for better performance
        objs = get_objects(interpreter, args.threshold)[: args.top_k]
        end_time = time.monotonic()
        text_lines = [
            "Inference: {:.2f} ms".format((end_time - start_time) * 1000),
            "FPS: {} fps".format(round(next(fps_counter))),
            "Objects detected: {}".format(len(objs)),
        ]
        print(' '.join(text_lines))

        # if FIRST_RUN == True and len(objs) > 16:
        #     jsonObjs = json.dumps(objs)
        #     with open(
        #         "/home/mendel/cinito_vision/cup_positions.json", "w", encoding="utf-8"
        #     ) as f:
        #         json.dump(jsonObjs, f, ensure_ascii=False, indent=4)

        # # Load reference list of cups
        # f = open("/home/mendel/cinito-vision/resources/cup_positions.json")
        # data = json.load(f)
        # cup_reference_list = json.loads(data)
        # # print("Reference cups loaded...")

        # cup_bbox = []
        # for cup_reference in cup_reference_list:
        #     if cup_reference[0] == 1:
        #         cup_bbox.append(cup_reference[2])

        # cup_bbox = sorted_bbox(cup_bbox)

        # Get detected cups
        # print("Number of detected Objects:", len(objs))
        cups = []
        for cup in objs:
            if cup[0] == 1:
                cups.append(cup[2])

        cup_list = []
        if len(cups) > 0:
            for cup in cups:
                cup_list.append(list(cup))
            cup_list = sorted_bbox(cup_list)

            pos_list = []
            for cup in cup_list:
                pos = centerIsInside(cup, cup_bbox)
                pos_list.append(pos)
            positive_values = [x for x in pos_list if x >= 0]
            if len(positive_values) > 0:
                minimum_positive = min(positive_values)
        else:
            minimum_positive = -1

        # DATA = struct.pack("i", minimum_positive)
        # DATA = bytearray(DATA)
        # # DATA = minimum_positive
        # client.publish(TOPIC, DATA, qos=QOS)
        # client.publish(TOPIC_INT, minimum_positive, qos=QOS)
        # client.publish(TOPIC_COUNT, (len(objs) - 1), qos=QOS)
        # # print("Next Cup: ", minimum_positive)
        time.sleep(1)
        return generate_svg(src_size, inference_box, objs, labels, text_lines)

    result = gstreamer.run_pipeline(
        user_callback,
        src_size=(CAM_W, CAM_H),
        appsink_size=inference_size,
        videosrc=args.videosrc,
        videofmt=args.videofmt,
        headless=True,
    )

def get_reference_positions(args):
    try:
        print("Loading reference positions {}".format(FILE_PATH))
        f = open(FILE_PATH)
        data = json.load(f)
        cup_reference_list = json.loads(data)
        cup_bbox = []
        for cup_reference in cup_reference_list:
            if cup_reference[0] == 1:
                cup_bbox.append(cup_reference[2])

        cup_bbox = sorted_bbox(cup_bbox)
        return cup_bbox, args.init


    except FileNotFoundError:
        print("File not found. A new reference file will be created.")
        args.init = True
        return None, args.init

# Callback functions for connection and message events
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT broker")
    else:
        print("Connection failed")


def on_disconnect(client, userdata, rc):
    if rc != 0:
        print("Unexpected disconnection from MQTT broker")
        client.publish(TOPIC, "Connection broken")
        while not client.is_connected():
            try:
                client.reconnect()
            except:
                print("Error when reconnecting. Next attempt in 5 seconds...")
                time.sleep(5)

if __name__ == "__main__":
    main()