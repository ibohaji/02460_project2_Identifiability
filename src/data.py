from torchvision import datasets, transforms
from src.utils import subsample 
from configs import TrainingConfig
import torch 




train_tensors = datasets.MNIST(
    "data/", train=True, download=True,
    transform=transforms.Compose([transforms.ToTensor()])  # Convert images to tensors
)

test_tensors = datasets.MNIST(
    "data/", train=False, download=True,
    transform=transforms.Compose([transforms.ToTensor()])  # Convert images to tensors
)
train_data = subsample(
    train_tensors.data, train_tensors.targets,
    TrainingConfig.num_train_data, TrainingConfig.num_classes
)
test_data = subsample(
    test_tensors.data, test_tensors.targets,
    TrainingConfig.num_train_data, TrainingConfig.num_classes
)

batch_size=32
mnist_train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size, shuffle=True
)
mnist_test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=batch_size, shuffle=False
)
