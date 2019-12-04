import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


######################################## 
#  Loading and normalizing CIFAR10

transform = transforms.Compose([transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

######################################## 
'''
# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
'''
######################################## 

class Net(nn.Module):

    def __init__(self):
	super(Net, self).__init__()
	
	# we have an RGB image, thus 3 input channels, 12 kernels, 3x3 kernel
	# 2 padding layers for the first layer; define 3 layers in total
	self.convLayer1 = nn.Conv2d(3,12,kernel_size=3,padding=2)
	self.convLayer2 = nn.Conv2d(12,16,kernel_size=3,padding=2)
	self.convLayer3 = nn.Conv2d(16,16,kernel_size=3,padding=2)
	# define max 2x2 pooling
	self.pool = nn.MaxPool2d(2,2)
	# define feed forward networks
	self.fc1 = nn.Linear(16*5*5,120)
	self.fc2 = nn.Linear(120,84)
	self.fc3 = nn.Linear(84,10)
	
    def forward(self, x):
	# pass the input through the convolution layer, then through a relu
	# activation function, followed by max pooling for each layer 
	x = self.pool(F.relu(self.convLayer1(x)))
	x = self.pool(F.relu(self.convLayer2(x)))
	x = self.pool(F.relu(self.convLayer3(x)))
	# flatten the image
	x = x.view(-1,16*5*5)
	x = F.relu(self.fc1(x))
	x = F.relu(self.fc2(x))
	x = F.relu(self.fc3(x))
	return x

# create an instance of the cnn
net = Net()

# train on GPU if possible
'''
use_cuda = True
if use_cuda and torch.cudais_available():
    net.cuda()
'''

######################################## 

# Define a Loss function and optimizer
# cross entropy loss
lossFunction = nn.CrossEntropyLoss()
# stochastic gradient descent optimizer
optimizer = optim.SGD(net.parameters(), lr=0.0005, momentum=0.975)

######################################## 
 #Train the network
for epoch in range(3):  
    # 1- forward pass +otimize + backward pass
    # 2- calculate and print loss

    run_loss = 0.0
    for i,data in enumerate(trainloader,0):
	inputs,labels = data

	optimizer.zero_grad()
	# forward pass
	outputs = net(inputs)
	# compute error and back propogate
	loss = lossFunction(outputs,labels)
	loss.backward()
	# update weights
	optimizer.step()
	run_loss += loss.item()

	# print every 2000 batches of training
	if i % 2000 == 1999:
	    print('[epoch, batch]:[%d, %5d], loss: %.3f' %
	          (epoch + 1, i + 1, run_loss / 2000))
	    run_loss = 0.0	
print('\nTraining Complete')

########################################

# Save the trained model
loc = './model'
torch.save(net.state_dict(),loc)

########################################

# Print Confusion matrix for the test set
# declare variables to measure accuracy
correct = 0
total = 0
conf_mat = torch.zeros(10,10)
with torch.no_grad():
    # test network on all test images
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
	# continuously build the confusion matrix
	for t,p in zip(labels.view(-1), predicted.view(-1)):
	    conf_mat[t.long(), p.long()] += 1

# print total accuracy and confusion matrix
print('Network Prediction Accuracy on 10000 test images: %d %%' % (
    100 * correct / total))
print(conf_mat)

######################################## 
######################################## 

