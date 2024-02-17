from torch import nn


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.LeNet = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(5, 5)),
            nn.Flatten(),
            nn.Linear(221880, 84),
            nn.ReLU(),
            nn.Linear(84, 29)
        )

        self.LeNet_Gray = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(5, 5)),
            nn.Flatten(),
            nn.Linear(221880, 84),
            nn.ReLU(),
            nn.Linear(84, 29)
        )

        self.LeNet_Tiny = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(5, 5)),
            nn.Flatten(),
            nn.Linear(38880, 84),
            nn.ReLU(),
            nn.Linear(84, 29)
        )

        self.AlexNet = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(11, 11), stride=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5, 5), padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=192, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            nn.Dropout(),
            nn.Flatten(),
            nn.Linear(4800, 150),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(150, 150),
            nn.ReLU(),
            nn.Linear(150, 29)
        )

        self.AlexNet_Tiny = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(11, 11), stride=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5, 5), padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=192, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            nn.Dropout(),
            nn.Flatten(),
            nn.Linear(768, 150),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(150, 150),
            nn.ReLU(),
            nn.Linear(150, 29)
        )

        self.AlexNet_Gray = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=96, kernel_size=(11, 11), stride=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5, 5), padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=192, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            nn.Dropout(),
            nn.Flatten(),
            nn.Linear(4800, 150),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(150, 150),
            nn.ReLU(),
            nn.Linear(150, 29)
        )

    def forward(self, x):
        for layer in self.LeNet_Gray:
            x = layer(x)
            #print(x.size())
        return x

