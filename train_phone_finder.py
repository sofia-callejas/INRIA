# Standard Python libraries and general operations
import os
import shutil
from pathlib import Path
from dataclasses import dataclass
from IPython.display import display
import sys

# Libraries for data manipulation and image processing
import cv2

# torchvision libraries for computer vision models
from torchvision import transforms

# Created libraries
from Preprocessing.Transforms import Resize, RandomHorizontalFlip, RandomVerticalFlip, RandomTranslate, Create_BBox
from Preprocessing.DataCustom import PhoneImagesDataset

# Specific library from Ultralytics (YOLO)
from ultralytics import YOLO

def train_phone_finder(path):
    """This function is used to train the model

    Args:
        path (str): The path to the JPEG image to be tested.

    Returns:
        None
    """

    #Preprocessing
    dataset = PhoneImagesDataset(path,transform= transforms.Compose([Resize((256,256)),RandomHorizontalFlip(0.5),RandomVerticalFlip(0.5),RandomTranslate(0.5),Create_BBox(40,3)]))

    #DatasSplitting
    # Define the root folder for the dataset
    root_folder = Path("./datasets")

    root_folder = Path("./datasets")
    images_folder = root_folder / "images"
    labels_folder = root_folder / "labels"
    img_train_path = images_folder / "train"
    img_test_path = images_folder / "test"
    labels_train_path = labels_folder / "train"
    labels_test_path = labels_folder / "test"

    img_train_path.mkdir(parents=True, exist_ok=True)
    img_test_path.mkdir(parents=True, exist_ok=True)
    labels_train_path.mkdir(parents=True, exist_ok=True)
    labels_test_path.mkdir(parents=True, exist_ok=True)

     # This time we use all the data
    for image in dataset:
        name = image['name']
        label_path = labels_train_path / f"{name.split('.')[0]}.txt"
        label_str = f"0 {image['x']} {image['y']} {image['w']} {image['h']}"
        with open(label_path, 'w') as file:
            file.write(label_str)
        output_path = os.path.join(img_train_path, name)
        cv2.imwrite(output_path, image["image"])

    model = YOLO('yolov8n.pt')

    # Training the model
    summary = model.train(data='dataset.yaml', epochs=20, imgsz=256, device='cpu')

    # Save the trained model
    model.save('find_phone.pt')

    # Clean up temporary dataset folders
    shutil.rmtree(root_folder)

def main():
    """ This function is used to run the program. """
    train_phone_finder(sys.argv[1])

if __name__ == "__main__":
    main()

