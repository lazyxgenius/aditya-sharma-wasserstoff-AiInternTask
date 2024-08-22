# models/segmentation_model.py
import setuptools
import tensorflow as tf
import tensorflow_hub as hub
import numpy
from PIL import Image


# Load the Mask R-CNN model from TensorFlow Hub
def load_segmentation_model():

    model = hub.load("https://www.kaggle.com/models/tensorflow/faster-rcnn-inception-resnet-v2/TensorFlow2/1024x1024/1")
    return model


def segment_image(model, image_tensor):
    # Run the model to get the predictions
    output = model(image_tensor)

    # Extract scores, bounding boxes, and classes
    unfiltered_classes = output['detection_classes'][0].numpy()
    unfiltered_boxes = output['detection_boxes'][0].numpy()
    unfiltered_scores = output['detection_scores'][0].numpy()

    # Set a confidence threshold
    confidence_threshold = 0.5

    # Filter out boxes with low confidence scores using boolean masking
    high_confidence_mask = unfiltered_scores > confidence_threshold

    boxes = unfiltered_boxes[high_confidence_mask]
    scores = unfiltered_scores[high_confidence_mask]
    classes = unfiltered_classes[high_confidence_mask]

    # Print debugging information to verify extraction
    print("Boxes in segment_image function:", boxes)
    print("Classes in segment_image function:", classes)
    print("Scores in segment_image function:", scores)

    return scores, boxes, classes


