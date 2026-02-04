# 🧠 MNIST Image Generation using GAN (PyTorch)

This project implements a **Generative Adversarial Network (GAN)** from scratch using **PyTorch** to generate handwritten digit images similar to the MNIST dataset.

The primary goal of this project is **learning and understanding GAN fundamentals**, including adversarial training, data pipelines, and GPU-based model training.

---

## 📌 Project Overview

A Generative Adversarial Network (GAN) consists of two neural networks:

- **Generator**  
  Learns to generate fake images from random noise.

- **Discriminator**  
  Learns to distinguish between real images and generated (fake) images.

Both networks are trained together in an adversarial setup, where the Generator improves by trying to fool the Discriminator.

This implementation uses **fully connected (MLP-based) neural networks**, focusing on conceptual clarity rather than high-level abstractions.

---

## 🏗️ Model Architecture

### 🔹 Generator
- Input: Random noise vector (100 dimensions)
- Output: Flattened image (784 values → 28×28)
- Layers:
  - Linear → ReLU
  - Linear → Tanh

### 🔹 Discriminator
- Input: Flattened image (784 values)
- Output: Probability (real or fake)
- Layers:
  - Linear → LeakyReLU
  - Linear → Sigmoid

> ⚠️ This is an **MLP-based GAN**, not a CNN/DCGAN.

---

## 🧪 Dataset

- **MNIST (PNG format)**
- Grayscale handwritten digit images
- Image size: 28 × 28
- Loaded using `torchvision.datasets.ImageFolder`

### Preprocessing:
- Grayscale conversion
- Image → Tensor
- Normalization to range `[-1, 1]`

---

## 🚀 Training Details

- Framework: PyTorch
- Loss Function: Binary Cross Entropy Loss (BCELoss)
- Optimizer: Adam
- Learning Rate: `0.0002`
- Batch Size: `64`
- Training Type: Unsupervised Learning
- Hardware: GPU (CUDA-enabled)

---

## 📈 Training Progress (Visual Results)

The following images show how the Generator improves over training epochs.

### Epoch 1
Initial random noise, no recognizable structure.

![Epoch 1]
<img width="712" height="756" alt="Screenshot 2026-02-04 140454" src="https://github.com/user-attachments/assets/934f8735-4e67-4908-8a5f-6e0e588c6981" />


---

### Epoch 10
Basic strokes start appearing.

![Epoch 10]
<img width="743" height="803" alt="Screenshot 2026-02-04 152533" src="https://github.com/user-attachments/assets/8b05c150-877f-4ceb-8778-63b8a6d7386f" />



---

### Epoch 30
Digit shapes become more recognizable.

![Epoch 30]
<img width="746" height="780" alt="Screenshot 2026-02-04 155644" src="https://github.com/user-attachments/assets/cf7f99bc-aa6d-4512-af6c-e322bc9ce0da" />


---

### Epoch 80
Clear handwritten digits with improved consistency.

![Epoch 80]
<img width="600" height="600" alt="Figure_1" src="https://github.com/user-attachments/assets/bd818b9e-76cf-485a-b8e4-06bc2e47853e" />


---

## 🧠 Key Learnings

- Implemented a GAN training loop from scratch in PyTorch
- Understood adversarial learning dynamics
- Learned GPU (CUDA) usage and device management
- Debugged CPU/GPU mismatch issues
- Observed GAN convergence and instability behavior
- Built a complete deep learning data pipeline

---

## ⚠️ Limitations

- MLP-based GAN does not preserve spatial structure well
- Image quality is limited compared to CNN-based GANs
- Not suitable for complex or high-resolution image generation

---

## 🔮 Future Improvements

- Upgrade to **DCGAN (CNN-based GAN)**
- Implement Conditional GANs
- Experiment with higher-resolution datasets
- Improve training stability

---

## 🛠️ Technologies Used

- Python
- PyTorch
- Torchvision
- CUDA (GPU Acceleration)
- Matplotlib

---

## 📌 Conclusion

This project was built to gain a **deep, hands-on understanding of GANs and PyTorch**, focusing on implementation details, training behavior, and GPU usage rather than model performance alone.
