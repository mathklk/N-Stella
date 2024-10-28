#!/usr/bin/python3
from argparse import ArgumentParser
import cv2

parser = ArgumentParser(description='Count stars in an image of the night sky.')
parser.add_argument('image', help='Path to the image file.')
parser.add_argument('--show', action='store_true', help='Show the image.')
args = parser.parse_args()

# Load the image
image = cv2.imread(args.image)

# denoise
image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

# Convert to grayscale
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# binary threshold
_, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Count pixel groups
num_labels, _, _, _ = cv2.connectedComponentsWithStats(image, connectivity=8)

print(num_labels - 1)

if args.show:
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()