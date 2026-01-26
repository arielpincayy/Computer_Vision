# Computer Vision Notebooks

Welcome to the GitHub repository for the Computer Vision course. This repository is designed to provide you with hands-on experience and in-depth understanding of fundamental AI topics. The notebooks include both coding exercises and project-based activities, and were created using Python 3 as the interpreter.


## Getting Started

1. Clone this repository to your local machine:  

   ```
   git clone https://github.com/eugeniomorocho/Computer_Vision.git
   ```

2. Navigate to the specific Notebook's directory:  

   ```
   cd Computer_Vision/NOTEBOOK_x/
   ```
   
3. Follow the instructions in the file for each week's lab.

4. To update your local fork to the newest commit, execute:

   ```
   git fetch 
   ```



## Requirements

- Python 3.x as the interpreter
- Additional dependencies specified in each week's lab instructions

## Support and Feedback

If you encounter any issues or have suggestions for improvement, please [open an issue](https://github.com/eugeniomorocho/Computer_Vision/issues). We appreciate your feedback!

## Course Structure

| **Unit** | **Topics & Concepts** | **Tools & Frameworks** | **Lab / Deliverable** | **Complementary Online Courses**
|:--|:--|:--|:--|:--|
| **Unit 1: Foundations of Computer Vision & Convolutions** | - What is modern Computer Vision?<br>- Images as tensors<br>- Filtering intuition<br>- Convolutions in images vs CNNs | Python, NumPy, OpenCV | Implement image filters manually and visualize convolution effects | [Image Processing in Python (DataCamp)](https://app.datacamp.com/learn/courses/image-processing-in-python)|
| **Unit 2: CNNs for Image Classification** | - Neural networks for vision<br>- Convolutional layers, pooling<br>- Training pipeline<br>- Evaluation metrics | PyTorch | Train a CNN for image classification |
| **Unit 3: Transfer Learning & Model Improvement** | - Overfitting & regularization<br>- Data augmentation<br>- Pretrained CNNs<br>- Fine-tuning vs feature extraction | PyTorch, torchvision | Transfer learning using ResNet or MobileNet |
| **Unit 4: Object Detection (Midterm Unit)** | - Classification vs detection<br>- Bounding boxes & IoU<br>- YOLO architecture<br>- Dataset annotation | Ultralytics YOLO, Roboflow <br> [**1. Drawing a Bounding Box with OpenCV.ipynb**](https://github.com/eugeniomorocho/Computer_Vision/blob/main/UC.06%20Object%20Detection%20(YOLO%20%2B%20Roboflow)/Object%20Detection/1.%20Drawing%20a%20Bounding%20Box%20with%20OpenCV.ipynb)| Train an object detector on a custom dataset with [Roboflow](https://roboflow.com) |
| **Unit 5: Image Segmentation & Pose Estimation** | - Semantic vs instance segmentation<br>- Encoder–decoder architectures<br>- Human pose estimation basics | PyTorch, MediaPipe | Segmentation **or** pose estimation mini-project |
| **Unit 6: Tracking & Video Analysis** | - Detection vs tracking<br>- Classical trackers (KCF, CSRT)<br>- Tracking-by-detection | OpenCV | Object tracking in video streams |
| **Unit 7: Model Deployment & Edge AI** | - Inference vs training<br>- ONNX<br>- APIs for vision models<br>- Edge inference and hardware constraints | ONNX Runtime, FastAPI, Jetson Nano | Deploy a trained model as an API **or** run inference on Jetson Nano |
| **Unit 8: Cloud & Modern Vision AI + Final Project Presentations** | - Vision APIs in the cloud<br>- Vision Transformers & SAM (conceptual)<br>- Ethics and real-world deployment | Cloud vision services (overview) | Use a cloud or foundation vision model for inference and compare results |




### Unit 1: Introduction to Computer Vision
OpenCV <br>
DataCamp: Introduction to Python

### Unit 2: Image Filtering and Convolutions
OpenCV 

### Unit 3: Naural Networks and CNN (Image Classification)
PyTorch

### Unit 4: Transfer Learning
PyTorch

### Unit 5: Face Detection
OpenCV (Haar / DNN Module)\
PyTorch

### Unit 6: Object Detection (Ultralytics YOLO + Roboflow)
PyTorch

| Drawing a Bounding box | Notebook |\
| YOLO on an image | Notebook |


[🖥️ Ultralytics website](https://www.ultralytics.com)<br>
[🌎 Roboflow website](https://roboflow.com)

### Unit 7: Image Segmentation
PyTorch

### Unit 8: 3D Reconstruction (drone imagery)

### 9. Pose Estimation

### # Mobile Computer Vision (on the edge)

MediaPipe <br>
[Flutter](https://flutter.dev/) <br>
Android/iOS/Web

### Model Deployment

Running Model with ONNX Runtime <br>
Creating API with FastAPI

### 10. Tracking
OpenCV (KCF, CSRT)

### UC.11 Generative AI

### NVIDIA TAO Toolkit and DeepStream (Docker + Jetson Nano 2GB Developer Kit) ONNX? 
Sagemaker de NVIDIA, pero está en la siguiente unidad?

### UC.13 Computer Vision on the Cloud (AWS REkognition, Lookout for Vision, and SageMaker)

## Contenido

| Semana | Notebook                          | Temas| 
| :---:  | :---------------------------------------------   | :---------------------------------------------   |
| 1      | [Introducción a las redes neuronales](https://marsgr6.github.io/presentations/rnas_html/S1/S1_intro_ann.html) | Introducción a las redes neuronales: desde la neurona de McCulloch-Pitts hasta el perceptrón multicapa, explorando los fundamentos del procesamiento neuronal artificial. |

## Bibliography

### Primary Books

[1] Torralba, A., Isola, P., & Freeman, W. (2024). Foundations of Computer Vision. MIT Press. https://visionbook.mit.edu/

[2] Szeliski, R. (2022). Computer Vision: Algorithms and Applications (2nd ed.). Springer Cham. https://doi.org/https://doi.org/10.1007/978-3-030-34372-9 

[3] Ayyadevara, V. K., & Reddy, Y. (2024). Modern Computer Vision with PyTorch: A practical roadmap from Deep Learning fundamentals to advanced applications and Generative AI (2nd ed.). Packt Publishing Ltd. https://www.packtpub.com/en-mt/product/modern-computer-vision-with-pytorch-9781803240930 
<br>
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/PacktPublishing/Modern-Computer-Vision-with-PyTorch-2E)

[4] Shanmugamani, R. (2018). Deep Learning for Computer Vision: Expert techniques to train advanced neural networks using TensorFlow and Keras (1st ed.). Packt Publishing. https://www.packtpub.com/en-us/product/deep-learning-for-computer-vision-9781788295628 
<br>
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/packtpublishing/deep-learning-for-computer-vision)

[5] Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning (1st ed.). The MIT Press. https://www.deeplearningbook.org

### Complementary Books

[6] Elgendy, M. (2020). Deep Learning for Vision Systems. Manning Publications Co. https://www.manning.com/books/deep-learning-for-vision-systems 

[7] Prince, S. J. D. (2012). Computer Vision: Models, Learning and Inference. Cambridge University Press. https://www.cambridge.org/ca/universitypress/subjects/computer-science/computer-graphics-image-processing-and-robotics/computer-vision-models-learning-and-inference

[8] Zhang, A., Lipton, Z. C., Li, M. U., & Smola, A. J. (2023). Dive into Deep Learning. Cambridge University Press. https://D2L.ai
<br>
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/d2l-ai/d2l-en)

[9] Chollet, F. (2026). Deep Learning with Python (3rd ed.). Manning Publications. https://deeplearningwithpython.io 

### Research Papers

[10] Vaswani, A., Brain, G., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention Is All You Need. NIPS’17: Proceedings of the 31st International Conference on Neural Information Processing Systems, 6000–6010. https://doi.org/10.48550/arXiv.1706.03762

[11] Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., & Houlsby, N. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR 2021 - 9th International Conference on Learning Representations. https://arxiv.org/abs/2010.11929v2

[12] Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., & Zagoruyko, S. (2020). End-to-End Object Detection with Transformers. Lecture Notes in Computer Science (Including Subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics), 12346 LNCS, 213–229. https://doi.org/10.1007/978-3-030-58452-8_13

[13] SKirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A. C., Lo, W. Y., Dollár, P., & Girshick, R. (2023). Segment Anything. Proceedings of the IEEE International Conference on Computer Vision, 3992–4003. https://doi.org/10.1109/ICCV51070.2023.00371

### Online Resources

[14] [Stanford CS231N Deep Learning for Computer Vision 2025 (YouTube Playlist)](https://www.youtube.com/playlist?list=PLoROMvodv4rOmsNzYBMe0gJY2XS8AQg16)

[15] [Stanford Lecture Collection CNNs for Visual Recognition 2017](https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv)

[16] [NVIDIA Deep Learning Institute](https://www.nvidia.com/en-us/training/)

---
<br>
<p style="text-align: right; font-size:14px; color:gray;">
<b>Prepared by:</b><br>
Manuel Eugenio Morocho-Cayamcela, Ph.D.
</p>

<div style="text-align: right;">
  <img src="yt.png" alt="drawing" style="width: 100px;" />
</div>