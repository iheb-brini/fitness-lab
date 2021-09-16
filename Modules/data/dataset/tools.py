import torchvision
from torchvision import transforms
from torch.utils import data


def load_data_fashion_mnist(root, batch_size,
                            resize=None, num_worker=4):
    """Download the Fashion-MNIST dataset and then load it into memory."""
    operations = []
    if resize:
        operations.append(transforms.Resize(resize))

    operations.append(transforms.ToTensor())

    trs = transforms.Compose(operations)

    mnist_train = torchvision.datasets.FashionMNIST(
        root=root, train=True, transform=trs, download=True)

    mnist_test = torchvision.datasets.FashionMNIST(
        root=root, train=False, transform=trs, download=True)

    train_iter = data.DataLoader(mnist_train, batch_size,
                                 shuffle=True, num_workers=num_worker)

    test_iter = data.DataLoader(mnist_test, batch_size,
                                shuffle=False, num_workers=num_worker)

    return train_iter, test_iter
