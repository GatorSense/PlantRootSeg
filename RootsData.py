import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

class RootsDataset(Dataset):
    
    def __init__(self, root, train=True, mode='RGB', img_transform=None, 
                 label_transform=None):     
        """
        Initialize the train/test dataset for Unet training.

        Parameters
        ----------
        root : str,
            The data path for root images.
        train : bool, optional
            Specify whether the subset will be used for training. 
            The default is 'True'.
        mode : str, options: 'RGB' or 'gray'.
            The mode of input images. 
            The default is 'RGB'.
        img_transform : pytorch transform functions, optional
            The pre-processing functions for input images using Pytorch Transformation. 
            The default is None.
        label_transform : pytorch transform functions, optional
            The pre-processing functions for image labels using Pytorch Transformation. 
            The default is None.

        """
        if not isinstance(train, bool):
            raise ValueError('Variable \'train\' should be boolean')
        
        if not isinstance(mode, str) or mode not in ('RGB', 'gray'):
            ValueError('Variable \'mode\' should be RGB or gray')
            
        self.root = root
        self.mode = mode
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.files = []
        
        imgset_dir = os.path.join(self.root, 'Images')
        
        for file in os.listdir(imgset_dir):
            img_name = os.path.splitext(file)[0]
            label_name = 'GT_' + img_name + '.png'
            img_file = os.path.join(imgset_dir, file)
            label_file = os.path.join(self.root, 'GT', label_name)
            self.files.append({
                    "img": img_file,
                    "label": label_file
                    })
        # split train/test subset by sampling every 10 images. Randomly selection
        # is not used because the images of different depths and dates vary a lot.
        # This data split method can be modified based on your own need.
        self.data_idx = np.arange(int(len(self.files)))
        self.test_idx = self.data_idx[::10]
        self.train_idx = np.setdiff1d(self.data_idx, self.test_idx)
        
        if train:
            self.files = [self.files[i] for i in self.train_idx]
        else:
            self.files = [self.files[i] for i in self.test_idx]
 
                
    def __len__(self):     
        return len(self.files)
    
    def __getitem__(self, index):
        datafiles = self.files[index]
        img_file = datafiles["img"]
        
        if self.mode == 'RGB':
            img = Image.open(img_file).convert('RGB')
        if self.mode == 'gray':
            img = Image.open(img_file).convert('L')
            img = img.convert('RGB')
                    
        label_file = datafiles["label"]
        label = Image.open(label_file).convert("P")
        
        if self.img_transform is not None:
            img = self.img_transform(img)
            
        if self.label_transform is not None:
            label = self.label_transform(label)  
       
        label = np.array(label)*255
        
        return img, label


