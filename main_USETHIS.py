import numpy as np
import pickle
import torch.optim as optim
import torch
from torchvision import datasets, transforms
from nnetsUSETHIS import Net
from train_loopUSETHIS import train_model
from make_dataset1 import MNISTDataset
from torch.utils.data import DataLoader
from predictions1 import get_prediction


BATCH_SIZE = 100
MOMENTUM = 0.9
LR = 1e-3
MAX_PATIENCE = 50

if __name__ == '__main__':

    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    # pretrain on MNIST

    net = Net()

    train_data = open('train_max_x', 'rb') #when train_max is available input it
    train_data = pickle.load(train_data)

    target = np.genfromtxt('train_y.csv', delimiter=',')
    target = np.int64(target)
    idx = 9 * train_data.shape[0] // 10

    print('training')
    max_mnist_train = DataLoader(MNISTDataset(train_data[:idx], target=target[:idx],
                                              transform=transforms.Compose([
                                                  transforms.ToPILImage(),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.1307,), (0.3081,))
                                              ])), batch_size=BATCH_SIZE, shuffle=True)

    max_mnist_test = DataLoader(MNISTDataset(train_data[idx:], target=target[idx:],
                                             transform=transforms.Compose([
                                                 transforms.ToPILImage(),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.1307,), (0.3081,))
                                             ])), batch_size=BATCH_SIZE, shuffle=True)

    optimizer_best = optim.SGD(net.parameters(), lr=LR, momentum=MOMENTUM)
    best_net = train_model(net, optimizer_best, max_mnist_train,
                           max_mnist_train, device, max_patience=MAX_PATIENCE)

    print('making prediction')
    test_data = open('test_max_x', 'rb')
    test_data = pickle.load(test_data)
    max_mnist_test = DataLoader(MNISTDataset(test_data,
                                              transform=transforms.Compose([
                                                  transforms.ToPILImage(),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.1307,), (0.3081,))
                                              ])), batch_size=BATCH_SIZE, shuffle=False)
    preds = get_prediction(best_net, max_mnist_test, device)
    np.savetxt('preds.csv', preds, delimiter=',')
