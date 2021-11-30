import argparse
import cv2

MEAN_SUBTRACTIONS = [103.939, 116.779, 123.680]

parser = argparse.ArgumentParser()
parser.add_argument("--image_path", type=str, dest="image_path", required=True, help="Path to load images")
parser.add_argument("--model_path", type=str, dest="model_path", required=True, help="Path to load models")
opt = parser.parse_args()

image_path = opt.image_path
model_path = opt.model_path

def post_process(image, mean_subtractions=MEAN_SUBTRACTIONS):

    image = image.reshape((3, image.shape[2], image.shape[3]))

    # add back in the mean subtraction
    image[0] += mean_subtractions[0]
    image[1] += mean_subtractions[1]
    image[2] += mean_subtractions[2]

    # scaling
    image = image.astype('float')/255.0

    #swap the order
    image = image.transpose(1, 2, 0)

    return image

# load image
image = cv2.imread(image_path)
height, width = image.shape[:2]

# load model
