import os
import torch
import pickle
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class MiniImageNet(Dataset):
    def __init__(self, root, mode='train', transform=None):
        split_tag = mode
        if mode == 'train': 
            split_tag = 'train_phase_train'
        split_file = 'miniImageNet_category_split_{}.pickle'.format(split_tag)
        # Read corresponding file
        with open(os.path.join(root, split_file), 'rb') as f:
            tmp = pickle.load(f, encoding='latin1')
        image = tmp['data']
        label = tmp['label']

        image_size = 80 
        image = [Image.fromarray(x) for x in image]
        
        min_label = min(label)
        label = [x - min_label for x in label]

        self.image = image 
        self.label = label 
        self.n_classes = max(self.label) + 1
        
        # Define preprocessed transform
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        default_transform = transforms.Compose([
            transforms.Resize(image_size),  
            transforms.RandomCrop(image_size, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        if transform is None: 
            self.transform = default_transform
        else:
            self.transform = transform 

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        image = self.image[i]
        label = self.label[i]
        image = self.transform(image)
        return image, label 
        
         
