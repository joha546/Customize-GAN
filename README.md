GAN (Generative Adversarial Network) for Handwritten Digit Generation

Overview

This repository contains a PyTorch implementation of a Generative Adversarial Network (GAN) for generating handwritten digit images, particularly focused on the MNIST dataset. The GAN consists of a Generator and a Discriminator, trained in an adversarial manner to produce realistic images.

Getting Started

Prerequisites

- Python 3
- PyTorch
- Matplotlib
- TQDM

Install dependencies using:

```bash
pip install torch matplotlib tqdm
```

### Training the GAN

Run the training script to start training the GAN:

```bash
python train_gan.py
```

This will train the GAN for a specified number of epochs. You can adjust hyperparameters in the script, such as the learning rate, batch size, and noise dimension.

### Generator Network

After training, you can use the trained Generator to generate new images. Example code is provided in the training script, and you can also run the following snippet:

```python
import torch
from gan_model import Generator
from utils import show_tensor_images

# Load trained Generator
G = Generator(noise_dim=64)
G.load_state_dict(torch.load("generator_weights.pth"))  # Replace with the actual path

# Generate images
batch_size = 16
noise = torch.randn(batch_size, 64)
generated_images = G(noise)

# Display generated images
show_tensor_images(generated_images)
```

Notes

- The weights of the Generator are saved to "generator_weights.pth" after training. Adjust the path accordingly when loading the Generator for generation.

- The training script uses the MNIST dataset, and the GAN is specifically designed for generating 28x28 grayscale images.

Feel free to experiment with the code, tweak hyperparameters, or adapt it to other datasets for different image generation tasks.
