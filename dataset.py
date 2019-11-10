from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F
import numpy as np
import torch
import pickle

class Adversarial_Dataset(Dataset):
    
    def __init__(self, root_dir, attack_name, transform = None):
        
        self.root_dir = root_dir
        self.attack = attack_name # FGSM, DF, SM(SaliencyMap)
        self.transform = transform
        image_name = ''
        label_name = ''
            
        image_name = root_dir + self.attack + '_adv_images.pickle'
        label_name = root_dir + self.attack + '_adv_label.pickle'

        
        with open (image_name, 'rb') as fp:
            self.images = pickle.load(fp)
            
        with open (label_name, 'rb') as fp:
            self.labels = pickle.load(fp)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        
        image = self.images[index].float()
        label = self.labels[index]
        
        if self.transform is not None:
            image = self.transform(image)
        
        return (image, label)