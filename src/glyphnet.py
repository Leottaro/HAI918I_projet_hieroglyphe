import os
import torch
from torch import nn, Tensor
from torchinfo import summary

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

class Glyphnet(nn.Module):
    def __init__(self, num_classes=40):
        super().__init__()

        self.num_classes = num_classes

        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                num_features=64,
                eps=1e-3,
                momentum=0.01,
            ),
            nn.MaxPool2d(
                kernel_size=3,
                stride=2,
                padding = 1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                num_features=64,
                eps=1e-3,
                momentum=0.01,
            ),
            nn.MaxPool2d(
                kernel_size=3,
                stride=2,
                padding = 1,
            ),
            nn.ReLU(),
        )

        self.block2 = nn.Sequential(
            # SeparableConv128, (input = 64)
            nn.Sequential(
                nn.Conv2d(
                    in_channels=64,
                    out_channels=64,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=64,
                    bias=False,
                ),
                nn.Conv2d(
                    in_channels=64,
                    out_channels=128,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                )
            ),
            nn.BatchNorm2d(
                num_features=128,
                eps=1e-3,
                momentum=0.01,
            ),
            nn.ReLU(),

            # SeparableConv128, (input = 128)
            nn.Sequential(
                nn.Conv2d(
                    in_channels=128,
                    out_channels=128,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=128,
                    bias=False,
                ),
                nn.Conv2d(
                    in_channels=128,
                    out_channels=128,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                )
            ),
            nn.BatchNorm2d(
                num_features=128,
                eps=1e-3,
                momentum=0.01,
            ),
            nn.MaxPool2d(
                kernel_size=3,
                stride=2,
                padding = 1,
            ),
            nn.ReLU(),
        )

        self.block3 = nn.Sequential(
            # SeparableConv128, (input = 128)
            nn.Sequential(
                nn.Conv2d(
                    in_channels=128,
                    out_channels=128,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=128,
                    bias=False,
                ),
                nn.Conv2d(
                    in_channels=128,
                    out_channels=128,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                )
            ),
            nn.BatchNorm2d(
                num_features=128,
                eps=1e-3,
                momentum=0.01,
            ),
            nn.ReLU(),

            # SeparableConv128, (input = 128)
            nn.Sequential(
                nn.Conv2d(
                    in_channels=128,
                    out_channels=128,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=128,
                    bias=False,
                ),
                nn.Conv2d(
                    in_channels=128,
                    out_channels=128,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                )
            ),
            nn.BatchNorm2d(
                num_features=128,
                eps=1e-3,
                momentum=0.01,
            ),
            nn.MaxPool2d(
                kernel_size=3,
                stride=2,
                padding = 1,
            ),
            nn.ReLU(),
        )

        self.block4 = nn.Sequential(
            # SeparableConv256, (input = 128)
            nn.Sequential(
                nn.Conv2d(
                    in_channels=128,
                    out_channels=128,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=128,
                    bias=False,
                ),
                nn.Conv2d(
                    in_channels=128,
                    out_channels=256,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                )
            ),
            nn.BatchNorm2d(
                num_features=256,
                eps=1e-3,
                momentum=0.01,
            ),
            nn.ReLU(),

            # SeparableConv256, (input = 256)
            nn.Sequential(
                nn.Conv2d(
                    in_channels=256,
                    out_channels=256,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=256,
                    bias=False,
                ),
                nn.Conv2d(
                    in_channels=256,
                    out_channels=256,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                )
            ),
            nn.BatchNorm2d(
                num_features=256,
                eps=1e-3,
                momentum=0.01,
            ),
            nn.MaxPool2d(
                kernel_size=3,
                stride=2,
                padding = 1,
            ),
            nn.ReLU(),
        )

        self.block5 = nn.Sequential(
            # SeparableConv256, (input = 256)
            nn.Sequential(
                nn.Conv2d(
                    in_channels=256,
                    out_channels=256,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=256,
                    bias=False,
                ),
                nn.Conv2d(
                    in_channels=256,
                    out_channels=256,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                )
            ),
            nn.BatchNorm2d(
                num_features=256,
                eps=1e-3,
                momentum=0.01,
            ),
            nn.ReLU(),

            # SeparableConv256, (input = 256)
            nn.Sequential(
                nn.Conv2d(
                    in_channels=256,
                    out_channels=256,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=256,
                    bias=False,
                ),
                nn.Conv2d(
                    in_channels=256,
                    out_channels=256,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                )
            ),
            nn.BatchNorm2d(
                num_features=256,
                eps=1e-3,
                momentum=0.01,
            ),
            nn.MaxPool2d(
                kernel_size=3,
                stride=2,
                padding = 1,
            ),
            nn.ReLU(),
        )

        self.block6 = nn.Sequential(
            # SeparableConv512, (input = 256)
            nn.Sequential(
                nn.Conv2d(
                    in_channels=256,
                    out_channels=256,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=256,
                    bias=False,
                ),
                nn.Conv2d(
                    in_channels=256,
                    out_channels=512,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                )
            ),
            nn.BatchNorm2d(
                num_features=512,
                eps=1e-3,
                momentum=0.01,
            ),
            nn.ReLU(),
            # Global avarage pooling
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(start_dim=1),
            nn.Dropout(p=0.15),
            nn.Linear(in_features=512,out_features=num_classes,),
        )


    def forward(self, x: Tensor) -> Tensor:

        output: Tensor = self.block1(x)

        output = self.block2(output)

        output = self.block3(output)

        output = self.block4(output)

        output = self.block5(output)

        logits = self.block6(output)
        
        return logits


def main():
    model = Glyphnet(40).to(device)
    model.eval()

    # print(model)

    x: Tensor = torch.randn(1, 1, 75, 50, device=device,)

    summary(
        model,
        input_data=x,
        col_names=("input_size", "output_size", "num_params"),
        depth=3,
        mode="eval",
    )


if __name__ == "__main__":
    main() 