import torch
import torchvision
from torch import nn
from PIL import Image

rgb_mean = torch.tensor([0.485, 0.456, 0.406])
rgb_std = torch.tensor([0.229, 0.224, 0.225])


def image_preprocessing(img_path):
    trans_totensor = torchvision.transforms.ToTensor()
    trans_resize = torchvision.transforms.Resize((256, 256))
    trans_norm = torchvision.transforms.Normalize(mean=rgb_mean, std=rgb_std)
    preprocessing = torchvision.transforms.Compose([trans_totensor, trans_resize, trans_norm])
    img = Image.open(img_path).convert("RGB")
    img_size = img.size
    img = preprocessing(img)
    img = torch.reshape(img, (1, 3, 256, 256))
    return img, img_size


def image_postprocessing(img, img_size):
    trans_resize = torchvision.transforms.Resize((img_size[1], img_size[0]))
    trans_toPIL = torchvision.transforms.ToPILImage(mode="RGB")
    trans_compose = torchvision.transforms.Compose([trans_resize, trans_toPIL])
    img = torch.squeeze(img)
    img = torch.clamp(img.permute(1, 2, 0) * rgb_std + rgb_mean, 0, 1)
    img = img.permute(2, 0, 1)
    img = trans_compose(img)
    return img


content_image_path = "1.jpg"
trans_toPIL = torchvision.transforms.ToPILImage(mode="RGB")
content_img, content_img_size = image_preprocessing(content_image_path)
img1 = trans_toPIL(content_img.reshape((3, 256, 256)))
img1.show()
img = image_postprocessing(content_img, content_img_size)
img.show()
