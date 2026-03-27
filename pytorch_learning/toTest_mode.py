import torch
from PIL import Image
from torchvision import transforms
from model import CIFAR10Classifier

img_path = r"pytorch_learning\img\image.png"
image = Image.open(img_path)
print(image)
# 调整大小
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

img_tensor = transform(image)
print(img_tensor.shape)

img_tensor = img_tensor.unsqueeze(0)
print(img_tensor.shape)

model = CIFAR10Classifier()
model.load_state_dict(torch.load(r"E:\Desktop\PycharmProjects\Numpy&Pandas\cifar10_model_10.pth"))
model.eval()

model.eval()
with torch.no_grad():
    output = model(img_tensor)
print(output)
print(output.argmax(1))
