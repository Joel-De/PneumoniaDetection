import torch
from torchvision.models import resnet50


class PneumoniaDetectionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet50()
        self.resnet = torch.nn.Sequential(*list(self.resnet.children())[:-1])
        self.l1 = torch.nn.Linear(in_features=2048, out_features=1024)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(1024, 128),
            torch.nn.Linear(128, 128),
            torch.nn.Linear(128, 64),
            torch.nn.Linear(64, 2),
        )

    def forward(self, image):
        image = self.resnet(image)
        image = torch.flatten(image, 1)
        imageVector = self.l1(image)
        output = self.classifier(imageVector)
        return output


if __name__ == "__main__":
    model = PneumoniaDetectionModel()
    model(torch.zeros(1, 3, 256, 256), torch.zeros(1, 3))
