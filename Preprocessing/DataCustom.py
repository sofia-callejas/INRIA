
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2


class PhoneImagesDataset(Dataset):
    """Phone Images dataset"""

    def __init__(self, image_paths, transform=None):
        
        """
        Arguments:
            image_paths (string): labels.txt path
        """
        
        self.image_paths = image_paths
        
        self.transform = transform
        
        self.idx = 0
        
    def __iter__(self):
        
        """Returns the iterator object."""
        
        return self
    
    def __next__(self):
        
        """
        Retrieves the next item in the dataset.
        """
        
        self.idx += 1
        
        try:
            
            return self[self.idx-1]
        
        except IndexError:
            
            self.idx = 0
            
            raise StopIteration
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        
        """
        Retrieves a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: A dictionary containing image and label information.
        """
        
        data = np.genfromtxt(self.image_paths + "/labels.txt", delimiter=' ', dtype=None, encoding=None)
        
        image_filepath = self.image_paths + "/" + data[idx][0]
        
        # Read the image using OpenCV

        image = cv2.imread(image_filepath)
                
        # Extract coordinates from the data

        x =  data[idx][1]
        y =  data[idx][2]
        
        # Create a sample dictionary with image and coordinate information
        
        sample = {"name": data[idx][0], "image": image, "x": x , "y" : y , "h": 0 , "w" : 0}
        
        # Apply the transform if provided
        if self.transform:
            sample = self.transform(sample)
            
        return sample