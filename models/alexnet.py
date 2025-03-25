from torch import nn


class AlexNet(nn.Module):
    def __init__(self, num_classes=200):  # Tiny ImageNet has 200 classes
        super(AlexNet, self).__init__()

        # Define the layers in AlexNet
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # Adjust the size here for the Tiny ImageNet input size
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 1 * 1, 4096),  # 256 * 1 * 1 based on the new output size
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),  # num_classes=200 for Tiny ImageNet
        )

    def forward(self, x):
        x = self.features(x)  # Pass input through feature extraction layers
        x = x.view(
            x.size(0), 256 * 1 * 1
        )  # Flatten the output (adjusted for Tiny ImageNet)
        x = self.classifier(x)  # Pass through fully connected layers
        return x
