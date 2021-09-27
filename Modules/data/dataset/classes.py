import torchvision
from torchvision import transforms
from torch.utils import data


class FashionMNIST:
    r"""
    Create FashionMNIST dataset

    ---------
    Example 
    --------- ::
    

        dataset = FashionMNIST('./data', 32)
        train_iter, test_iter = dataset()
        x,y = next(iter(test_iter))
        print(x.shape)
        print(y.shape)
    """
    def __init__(self, root, batch_size,
                 resize=None, num_worker=4) -> None:
        self.root = root
        self.batch_size = batch_size
        self.trs = self.preprocess(resize)
        self.num_worker = num_worker

    def preprocess(self, resize):
        operations = []
        if resize:
            operations.append(transforms.Resize(resize))

        operations.append(transforms.ToTensor())

        trs = transforms.Compose(operations)
        return trs

    def load(self):
        """Download the Fashion-MNIST dataset and then load it into memory."""
        mnist_train = torchvision.datasets.FashionMNIST(
            root=self.root, train=True, transform=self.trs, download=True)

        mnist_test = torchvision.datasets.FashionMNIST(
            root=self.root, train=False, transform=self.trs, download=True)

        train_iter = data.DataLoader(mnist_train, self.batch_size,
                                     shuffle=True, num_workers=self.num_worker)

        test_iter = data.DataLoader(mnist_test, self.batch_size,
                                    shuffle=False, num_workers=self.num_worker)

        return train_iter, test_iter

    def __call__(self):
        return self.load()


