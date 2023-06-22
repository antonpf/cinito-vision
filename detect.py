import argparse
import os
import sys
import time
import json
import operator

import paho.mqtt.client as mqtt

import pygame
import pygame.camera
from pygame.locals import *

from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference

TOPIC = "becherlager_test"
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
        "--threshold", type=float, default=0.55, help="classifier score threshold"
    )
    parser.add_argument(
        "--init",
        type=bool,
        default=False,
        help="initialises the reference positions of the individual cups",
    )
    args = parser.parse_args()

    # Create a MQTT client
    client = mqtt.Client()

    # Set up the callback functions
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect

    client.connect(BROKER_ADRESS, PORT)
    client.loop_start()
    detect_cups(args, client)
    client.loop_stop()


def detect_cups(args, client):
    pygame.init()
    pygame.font.init()
    font = pygame.font.SysFont("Arial", 20)

    pygame.camera.init()
    camlist = pygame.camera.list_cameras()

    cup_reference, args.init = get_reference_positions(args)

    print(cup_reference)

    labels, interpreter = get_model(args)

    inference_size = input_size(interpreter)

    camera = get_camera(CAM_W, CAM_H, camlist)

    display = get_display(CAM_W, CAM_H)

    red = pygame.Color(255, 0, 0)

    scale_x, scale_y = CAM_W / inference_size[0], CAM_H / inference_size[1]
    try:
        last_time = time.monotonic()
        while True:
            mysurface = camera.get_image()
            imagen = pygame.transform.scale(mysurface, inference_size)
            start_time = time.monotonic()
            run_inference(interpreter, imagen.get_buffer().raw)
            results = get_objects(interpreter, args.threshold)[: args.top_k]
            stop_time = time.monotonic()
            inference_ms = (stop_time - start_time) * 1000.0
            fps_ms = 1.0 / (stop_time - last_time)
            last_time = stop_time
            annotate_text = "Inference: {:5.2f}ms FPS: {:3.1f}".format(
                inference_ms, fps_ms
            )
            if args.init == True:
                print("Create new file ...")
                print("Save init file ...")
                args.init = False
            client.publish(TOPIC_COUNT, len(results), qos=QOS)
            print(results)
            draw_bbox(font, labels, red, scale_x, scale_y, mysurface, results)
            text = font.render(annotate_text, True, red)
            # print(annotate_text)
            mysurface.blit(text, (0, 0))
            display.blit(mysurface, (0, 0))
            pygame.display.flip()
    finally:
        camera.stop()

def get_reference_positions(args):
    try:
        with open(FILE_PATH) as file:
            print("Loading reference file: {}".format(FILE_PATH))
            cup_reference_positions = json.load(file)
            print("Test", cup_reference_positions)
            cup_reference = []
            for cup_reference_position in cup_reference_positions:
                if cup_reference_position[0] == 1:
                    cup_reference.append(cup_reference_position[2])

            cup_reference = sorted_bbox(cup_reference)
            return cup_reference, args.init

    except FileNotFoundError:
        print("File not found. A new reference file will be created.")
        args.init = True
        return None, args.init


def sorted_bbox(cup_bbox):
    cup_bbox = sorted(cup_bbox, key=operator.itemgetter(1))
    for i in range(4):
        start = i * 4
        end = i * 4 + 4
        row = cup_bbox[start:end]
        row = sorted(row, key=operator.itemgetter(0))
        cup_bbox[start:end] = row

    return cup_bbox


def draw_bbox(font, labels, red, scale_x, scale_y, mysurface, results):
    for result in results:
        bbox = result.bbox.scale(scale_x, scale_y)
        rect = pygame.Rect(bbox.xmin, bbox.ymin, bbox.width, bbox.height)
        pygame.draw.rect(mysurface, red, rect, 1)
        label = "{:.0f}% {}".format(
            100 * result.score, labels.get(result.id, result.id)
        )
        text = font.render(label, True, red)
        print(label, " ", end="")
        mysurface.blit(text, (bbox.xmin, bbox.ymin))


def get_display(cam_w, cam_h):
    try:
        display = pygame.display.set_mode((cam_w, cam_h), 0)
    except pygame.error as e:
        sys.stderr.write(
            "\nERROR: Unable to open a display window. Make sure a monitor is attached and that "
            "the DISPLAY environment variable is set. Example: \n"
            '>export DISPLAY=":0" \n'
        )
        raise e
    return display


def get_model(args):
    with open(args.labels, "r") as f:
        pairs = (l.strip().split(maxsplit=1) for l in f.readlines())
        labels = dict((int(k), v) for k, v in pairs)

    print("Loading {} with {} labels.".format(args.model, args.labels))

    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    labels = read_label_file(args.labels)
    return labels, interpreter


def get_camera(cam_w, cam_h, camlist):
    camera = None
    for cam in camlist:
        try:
            camera = pygame.camera.Camera(cam, (cam_w, cam_h))
            camera.start()
            print(str(cam) + " opened")
            break
        except SystemError as e:
            print("Failed to open {}: {}".format(str(cam), str(e)))
            camera = None
    if not camera:
        sys.stderr.write("\nERROR: Unable to open a camera.\n")
        sys, exit(1)
    return camera


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
