import torch
import torchvision
from torchvision import datasets, transforms

class DatasetConfig:
    def __init__(self, dataset_name, data_dir='data', batch_size=32, num_workers=4):
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = self.load_dataset()

    def load_dataset(self):
        if self.dataset_name == 'coco':
            transform = transforms.Compose([
                transforms.Resize((416, 416)),
                transforms.ToTensor(),
                # Add more transforms if needed
            ])
            dataset = datasets.CocoDetection(root=self.data_dir, annFile='annotations.json', transform=transform)
        elif self.dataset_name == 'imagenet':
            transform = transforms.Compose([
                transforms.Resize((416, 416)),
                transforms.ToTensor(),
                # Add more transforms if needed
            ])
            dataset = datasets.ImageFolder(root=self.data_dir, transform=transform)
        # Add more datasets as needed
        else:
            raise ValueError("Unsupported dataset. Please choose from 'coco', 'imagenet', etc.")
        
        return dataset

    def get_dataloader(self):
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, 
                                                 shuffle=True, num_workers=self.num_workers)
        return dataloader

def main():
    # Configure the dataset
    dataset_config = DatasetConfig(dataset_name='coco', data_dir='path_to_coco_dataset', batch_size=32, num_workers=4)
    
    # Get the dataloader
    dataloader = dataset_config.get_dataloader()
    
    # Iterate over the dataset
    for images, labels in dataloader:
        # Your training or inference code here
        pass

if __name__ == "__main__":
    main()