from torch import nn
from torchsummary import summary
from torchview import draw_graph

class StandardConv(nn.Module):
    def __init__(self, in_channels, out_channel, stride=1):
        super(StandardConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channel, kernel_size=3, stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x = nn.ReLU()(self.bn(self.conv(x)))
        return x


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channel, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        # Depthwise convolution
        self.conv_dw = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels)
        self.bn_dw = nn.BatchNorm2d(in_channels)

        # Pointwise convolution
        self.conv_pw = nn.Conv2d(in_channels, out_channel, kernel_size=1, stride=1, padding=0)
        self.bn_pw = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x = nn.ReLU()(self.bn_dw(self.conv_dw(x)))
        x = nn.ReLU()(self.bn_pw(self.conv_pw(x)))
        return x


class MobileNetV1(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(MobileNetV1, self).__init__()
        self.num_classes = num_classes

        self.model = nn.Sequential(
            StandardConv(in_channels, 32, stride=2),
            DepthwiseSeparableConv(32, 64, 1),
            DepthwiseSeparableConv(64, 128, 2),
            DepthwiseSeparableConv(128, 128, 1),
            DepthwiseSeparableConv(128, 256, 2),
            DepthwiseSeparableConv(256, 256, 1),
            DepthwiseSeparableConv(256, 512, 2),
            DepthwiseSeparableConv(512, 512, 1),
            DepthwiseSeparableConv(512, 512, 1),
            DepthwiseSeparableConv(512, 512, 1),
            DepthwiseSeparableConv(512, 512, 1),
            DepthwiseSeparableConv(512, 512, 1),
            DepthwiseSeparableConv(512, 1024, 2),
            DepthwiseSeparableConv(1024, 1024, 1),
            nn.AdaptiveAvgPool2d(1),
        )

        self.fc = nn.Linear(1024, self.num_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        return self.fc(x)


if __name__ == '__main__':
    model = MobileNetV1(in_channels=3, num_classes=10)
    summary(model, input_size=(3, 28, 28), device='cpu')

    model_graph = draw_graph(model, input_size=(1, 3, 28, 28), device='meta')

    # Save the graph using graphviz
    dot = model_graph.visual_graph
    dot.format = 'png'
    dot.attr(dpi='300')
    dot.render('mobilenetv1_graph')
