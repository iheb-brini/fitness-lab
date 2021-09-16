import torch
import torchvision
from torch import nn
from PIL import Image
import matplotlib.pyplot as plt

img_path="./bonobo.jpg"
test_img = Image.open(img_path)
plt.imshow(test_img); plt.show()
