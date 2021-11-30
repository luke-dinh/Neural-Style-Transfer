import argparse
import cv2

MEAN_SUBTRACTIONS = [103.939, 116.779, 123.680]

parser = argparse.ArgumentParser()
parser.add_argument("--image_path", type=str, dest="image_path", required=True, help="Path to load images")
parser.add_argument("--model_path", type=str, dest="model_path", required=True, help="Path to load models")
