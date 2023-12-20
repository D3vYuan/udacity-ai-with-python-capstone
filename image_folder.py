import torch
from torchvision import datasets, transforms

class ImageFolder:
    def __init__(self, train_data_dir, test_data_dir):
        self.train_data_dir = train_data_dir
        self.test_data_dir = test_data_dir
        self.load_train_data()
        self.load_test_data()
  
    def load_train_data(self):
        train_transforms = transforms.Compose([transforms.Resize(255),
                            transforms.RandomRotation(15),
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.Pad(4),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406],
                                                [0.229, 0.224, 0.225])]) 
        self.train_data = datasets.ImageFolder(self.train_data_dir, transform=train_transforms)

    def load_test_data(self):
        test_transforms = transforms.Compose([transforms.Resize(255),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406],
                                                [0.229, 0.224, 0.225])])
        self.test_data = datasets.ImageFolder(self.test_data_dir, transform=test_transforms)

    def load_batch_data(self, batch_size=64):
        train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=batch_size)
        return train_loader, test_loader

    def generate_class_to_index_map(self):
        return self.train_data.class_to_idx
