import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import json

class YOLODataset(Dataset):
    def __init__(self, image_folder, annotation_file, input_size=(416, 416), transform=None):
        self.image_folder = image_folder
        self.annotation_file = annotation_file
        self.input_size = input_size
        self.transform = transform
        self.labels = {}
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        for category in self.annotations['categories']:
            self.labels[category['id']] = category['name']
        
    def __len__(self):
        return len(self.annotations['images'])
    
    def __getitem__(self, idx):
        image_info = self.annotations['images'][idx]
        image_path = os.path.join(self.image_folder, image_info['file_name'])
        image = Image.open(image_path).convert('RGB')
        objects = []
        for ann in self.annotations['annotations']:
            if ann['image_id'] == image_info['id']:
                bbox = ann['bbox']
                label = ann['category_id']
                objects.append([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3], label])
        if self.transform:
            image, objects = self.transform(image, objects)
        return image, objects
    
    def collate_fn(self, batch):
        images = []
        targets = []
        for b in batch:
            images.append(b[0])
            targets.append(b[1])
        images = torch.stack(images, dim=0)
        return images, targets
    
class Rescale(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, image, objects):
        w, h = image.size
        new_w, new_h = self.output_size
        new_objects = []
        for obj in objects:
            x_min, y_min, x_max, y_max, label = obj
            x_min = x_min * new_w / w
            y_min = y_min * new_h / h
            x_max = x_max * new_w / w
            y_max = y_max * new_h / h
            new_objects.append([x_min, y_min, x_max, y_max, label])
        return image.resize((new_w, new_h), Image.BILINEAR), new_objects

class ToTensor(object):
    def __call__(self, image, objects):
        image = transforms.ToTensor()(image)
        targets = torch.zeros((len(objects), 6))
        for i, obj in enumerate(objects):
            targets[i, 1:] = torch.tensor(obj)
        return image, targets

# Define paths and parameters
image_folder = os.path.abspath('data/images')
annotation_file = os.path.abspath('data/annotations/instances_val2017.json')
input_size = (416, 416)

# Define transformations
transform = transforms.Compose([
    Rescale(input_size),
    ToTensor()
])

# Create dataset
dataset = YOLODataset(image_folder, annotation_file, input_size, transform)

# Create dataloader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=dataset.collate_fn)

for batch in dataloader:
    print(batch.shape)
