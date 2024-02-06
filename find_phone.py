# Libraries for data manipulation and image processing
import cv2

#Python libraries
import sys

# Specific library from Ultralytics (YOLO)
from ultralytics import YOLO

def find_phone(path):
    """This function is used to test the trained model.

    Args:
        path (str): The path to the JPEG image to be tested.

    Returns:
        None
    """
    
    # Load the trained YOLO model

    model = YOLO('best_find_phone.pt')
    # Read the test image
    image = cv2.imread(path)
    # Get the predictions from the YOLO model
    results = model(image)
    # Print the normalized coordinates of the detected phone
    print(tuple(results[0].boxes.xywhn[0][0:2].numpy()))

def main():
    """ This function is used to run the program. """
    find_phone(sys.argv[1])

if __name__ == "__main__":
    main()

