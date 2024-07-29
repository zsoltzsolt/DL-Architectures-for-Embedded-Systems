import argparse
import pickle
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import MobileNetV1
from tqdm import tqdm

def get_dataloader(dataset, batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    if dataset == 'cifar10':
        train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        num_classes = 10
    elif dataset == 'cifar100':
        train_data = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        test_data = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
        num_classes = 100
    elif dataset == 'mnist':
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        num_classes = 10
    elif dataset == 'fashion-mnist':
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
        num_classes = 10
    else:
        raise ValueError("Unsupported dataset")

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, num_classes


def train(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in tqdm(range(epochs)):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in tqdm(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')


def main():
    parser = argparse.ArgumentParser(description='Train MobileNetV1 on various datasets')
    parser.add_argument('dataset', type=str, choices=['cifar10', 'cifar100', 'mnist', 'fashion-mnist'],
                        help='Dataset to train on')
    parser.add_argument('epochs', type=int, help='Number of epochs to train for')
    args = parser.parse_args()

    train_loader, test_loader, num_classes = get_dataloader(args.dataset)

    model = MobileNetV1(in_channels=1, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train(model, train_loader, criterion, optimizer, args.epochs)

    with open('saved_models/mobilenetv1_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print('Model saved as mobilenetv1_model.pkl')


if __name__ == '__main__':
    main()