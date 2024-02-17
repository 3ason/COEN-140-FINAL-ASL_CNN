import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from NeuralNetwork import NeuralNetwork
from numpy import expand_dims

# change images to grayscale as possible experiement on outcomes

train_path = 'asl_alphabet_train_grayscale'

if train_path[-5:] == 'scale':
    cust_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1)
    ])
else:
    cust_transform = transforms.ToTensor()

dataset = datasets.ImageFolder(train_path, transform=cust_transform)


test_size = .2
num_samples = len(dataset)
train_test_split = int(num_samples * test_size)
batch_size = 64

# seed pytorch so that the permutations are the same each time program is run
torch.manual_seed(69)
seqs = torch.randperm(num_samples)

# separate data into training and testing
training_set = Subset(dataset, seqs[train_test_split:])
testing_set = Subset(dataset, seqs[:train_test_split])

# dataloader object created to facilitate batch loading 
training_loads = DataLoader(dataset=training_set, shuffle=True, batch_size=batch_size, num_workers=6)
testing_loads = DataLoader(dataset=testing_set, batch_size=batch_size, num_workers=6)

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


for X, y in testing_loads:
    # N = batchsize, C = num channels (1 = grayscale, 3 = rbg,...), H = height, W = width
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break


model = NeuralNetwork().to(device)
print(model)


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=.9)
#optimizer = torch.optim.Adam(model.parameters())


def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return test_loss

best_loss = 10
loss_history = []
epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(training_loads, model, loss_fn, optimizer, device)
    test_loss = test(testing_loads, model, loss_fn)
    loss_history.append(test_loss)

    if t > 7 and test_loss < best_loss:
        torch.save(model.state_dict(), "Best_LeNet_Model_Grayscale.pth")

print("Done!")

with open("LeNet_Loss_Values_Grayscale.txt", 'w') as f:
    for each in loss_history:
        f.write(str(each)+'\n')
