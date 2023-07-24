import torch
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor
from torch.utils import data
import torch.optim as optim
from UNetmodel import UNet
from RootsData import RootsDataset
import matplotlib.pyplot as plt
import argparse
from time import time
import os


# training function
def train(epoch):
    Unet.train()
    train_loss = torch.zeros(len(loader_train))
    for i, (img, label) in enumerate(loader_train):         
        img, label = img.to(device), label.to(device)
        output = Unet(img)
        loss = criterion(output, label)
        train_loss[i] = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print ("Epoch %d step %d, loss=%.8f" %(epoch, i, loss.item()))
    if args.outf is not None:
        if not os.path.exists(args.outf):
            os.mkdir(args.outf)
        if epoch % args.log_interval == 0:    
            torch.save(Unet.state_dict(), '%s/Unet_%d.pth' % (args.outf, epoch))
    return train_loss
    
# test function
def test():
    Unet.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (img, label) in enumerate(loader_test):
            img, label = img.to(device), label.to(device)
            output = Unet(img)
            test_loss += criterion(output, label).item()
    test_loss /= len(loader_test.dataset)
    print (" Test loss=%.5f" %(test_loss))
    return test_loss

# Training settings
parser = argparse.ArgumentParser(description='Unet for Roots Segmentation')
parser.add_argument('--data_path', type=str, default='./data', metavar='Data',
                    help='data path for training and testing')
parser.add_argument('--pretrain_model', type=str, default=None, metavar='PreM',
                    help='the path for pre-trained model')
parser.add_argument('--batch-size', type=int, default=2, metavar='Train Batch',
                    help='input batch size for training')
parser.add_argument('--test-batch-size', type=int, default=4, metavar='Test Batch',
                    help='input batch size for testing')
parser.add_argument('--epochs', type=int, default=2, metavar='E',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.8, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1, metavar='log_int',
                    help='how many batches to wait before logging training status')
parser.add_argument('--outf', type=str, default='./results',  help='output folder')

start = time()
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
#torch.manual_seed(args.seed)

input_transform = Compose([
    
    ToTensor(),

])

target_transform = Compose([

    ToTensor(),
 
])

# generate data loader for train and test set
loader_train = data.DataLoader(RootsDataset(args.data_path, train = True, mode = 'RGB',
                                    img_transform=input_transform,
                                    label_transform=target_transform),
                                batch_size=args.batch_size, shuffle=True, pin_memory=True)

loader_test = data.DataLoader(RootsDataset(args.data_path, train = False, mode = 'RGB',
                                    img_transform=input_transform,
                                    label_transform=target_transform),
                                batch_size=args.test_batch_size, shuffle=True, pin_memory=True)

# initialize Unet model
Unet = UNet(num_classes=1, depth=5).to(device)
if args.pretrain_model is not None:
    Unet.load_state_dict(torch.load(args.pretrain_model))
criterion = nn.BCEWithLogitsLoss().to(device)
optimizer = optim.SGD(Unet.parameters(), lr=args.lr, momentum=args.momentum)

# model training
test_loss = torch.zeros(args.epochs+1)
for epoch in range(1,args.epochs+1):
    E_start = time()
    train_loss = train(epoch)
    test_loss[epoch] = test()
    print(test_loss[torch.nonzero(test_loss)[:,0]].min(0))
    E_stop = time()
    print('==== Epoch Time ====', str(E_stop - E_start))
    
# plot the learning curve for this training
plt.figure()
plt.title('Learning Curve for Unet on Test Data')
plt.plot(test_loss[1:].numpy())
plt.xlabel('Epoch')
plt.ylabel('Test Loss')
stop = time()
print('==== Whole Time ====', str(stop-start))

