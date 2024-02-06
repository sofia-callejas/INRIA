import numpy as np
import cv2

class Resize(object):
    
        """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple): Desired output size. If tuple, output is
            matched to output_size.
    """
    
        def __init__(self ,size):
            self.size = size
    
        def __call__(self, sample):  
            
            # Resizes the image in the 'sample' dictionary to the specified size using OpenCV's cv2.resize function
            sample["image"] = cv2.resize(sample["image"], self.size)
            
            return sample
        
class RandomHorizontalFlip(object):
    
       
        """Apply random horizontal flip to the image and adjust bounding box coordinates in a sample.

    Args:
        p (float): Probability of applying the horizontal flip.
        
    """
    
        def __init__(self ,p):
            self.p = p
    
        def __call__(self, sample):  
            
            if np.random.rand() < self.p:
                
                # Flip the image horizontally

                sample["image"] = cv2.flip(sample["image"],1)
                
                 # Adjust bounding box coordinates
                    
                sample["x"] = 1 - sample["x"]
            
            return sample
        
class RandomVerticalFlip(object):
    
        """
         
    Apply random vertical flip to the image and adjust bounding box coordinates in a sample.

    Args:
        p (float): Probability of applying the vertical flip.
        
    """
    
        def __init__(self ,p):
            self.p = p
    
        def __call__(self, sample):  
            
            if np.random.rand() < self.p:
                
                # Flip the image vertically

                sample["image"] = cv2.flip(sample["image"],0)
                
                # Adjust bounding box coordinates
                    
                sample["y"] = 1 - sample["y"]
            
            return sample

class RandomTranslate(object):
    """
    Randomly translate the image and adjust bounding box coordinates in a sample.

    Args:
        p (float): Probability of applying the translation.
    """

    def __init__(self, p):
        self.p = p

    def __call__(self, sample):
        if np.random.rand() < self.p:
            
            max_translation = 0.2

            # Generate random translations in x and y
            translation_x = np.random.uniform(-max_translation, max_translation)

            translation_y = np.random.uniform(-max_translation, max_translation)

            # Create a translation transformation matrix
            translation_matrix = np.array([[1, 0, translation_x * sample["image"].shape[1]],
                                           [0, 1, translation_y * sample["image"].shape[0]]], dtype=np.float32)

            # Apply translation to the image using OpenCV's warpAffine function
            sample["image"] = cv2.warpAffine(sample["image"], translation_matrix, (sample["image"].shape[1], sample["image"].shape[0]))

            # Adjust bounding box coordinates based on the translation
            sample["x"] += translation_x
            sample["y"] += translation_y

        return sample

class Create_BBox(object):
    
        """Create a bounding box.

    Args:
        mean (int): Mean of a normal distribution for the height and width of the box.
        std (int): Standard deviation of a normal distribution for the height and width of the box.
    """
        
        def __init__(self ,mean,std):
            
            self.mean = mean
            self.std = std
            
        def __call__(self, sample):
            
            # Extract the image from the sample dictionary

            image = sample["image"]
            
            # Generate random bounding box dimensions based on a normal distribution

            h = np.random.normal(self.mean, self.std)/image.shape[1]
            w = np.random.normal(self.mean, self.std)/image.shape[0]
            
            # Add the generated dimensions to the sample dictionary
            
            sample['h'] = h
            sample['w'] = w
            
            return sample