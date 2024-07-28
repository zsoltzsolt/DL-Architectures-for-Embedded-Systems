# DL Architectures for Embedded Systems

Implementation and comparison of lightweight deep learning architectures optimized for embedded systems and mobile devices using PyTorch, with performance tracking via CometML.

## Objectives
- Implement and benchmark lightweight deep learning models using PyTorch.
- Utilize CometML to create a dashboard for tracking and comparing model performance metrics.
- Provide insights into the trade-offs between model complexity, performance, and efficiency on resource-constrained devices.

## Featured Architectures
1. **MobileNetV1**: Howard, A. G., et al. "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications." [Paper](https://arxiv.org/abs/1704.04861)
2. **MobileNetV2**: Sandler, M., et al. "MobileNetV2: Inverted Residuals and Linear Bottlenecks." [Paper](https://arxiv.org/abs/1801.04381)
3. **MobileNetV3**: Howard, A., et al. "Searching for MobileNetV3." [Paper](https://arxiv.org/abs/1905.02244)
4. **MobileNetV4**: Under research and development.
5. **EfficientNet**: Tan, M., and Le, Q. V. "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks." [Paper](https://arxiv.org/abs/1905.11946)
6. **ShuffleNet**: Zhang, X., et al. "ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices." [Paper](https://arxiv.org/abs/1707.01083)
7. **SqueezeNet**: Iandola, F. N., et al. "SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size." [Paper](https://arxiv.org/abs/1602.07360)
8. **NASNet**: Zoph, B., et al. "Learning Transferable Architectures for Scalable Image Recognition." [Paper](https://arxiv.org/abs/1707.07012)

## Repository Contents
- **Model Implementations**: PyTorch implementations of each architecture.
- **Training Scripts**: Scripts to train models on benchmark datasets.
- **Evaluation Metrics**: Scripts to evaluate model performance including accuracy, inference time, and model size.
- **CometML Integration**: Configuration files and scripts to log and visualize training metrics on CometML.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
