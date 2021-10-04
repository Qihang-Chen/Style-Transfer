import torch
from torch import nn
import torchvision
from PIL import Image
from torch.utils.tensorboard import SummaryWriter


class MyVgg16(nn.Module):
    def __init__(self):
        super(MyVgg16, self).__init__()
        self.seq = torchvision.models.vgg16(pretrained=True).features
        self.seq.requires_grad_(False)

    def forward(self, x, extract_list):
        output = {}
        count = 1
        for i in range(len(self.seq)):
            x = self.seq[i](x)
            if (i in extract_list):
                output["conv" + str(count)] = x
                count += 1
        return output


rgb_mean = torch.tensor([0.485, 0.456, 0.406]).cuda()
rgb_std = torch.tensor([0.229, 0.224, 0.225]).cuda()


def image_preprocessing(img_path):
    trans_totensor = torchvision.transforms.ToTensor()
    trans_resize = torchvision.transforms.Resize((512, 512))
    trans_norm = torchvision.transforms.Normalize(mean=rgb_mean, std=rgb_std)
    preprocessing = torchvision.transforms.Compose([trans_totensor, trans_resize, trans_norm])
    img = Image.open(img_path).convert("RGB")
    img_size = img.size
    img = preprocessing(img)
    img = torch.reshape(img, (1, 3, 512, 512))
    return img, img_size


def gram(feature_map):
    feature_map = torch.squeeze(feature_map)
    feature_map_flatten = torch.flatten(feature_map, 1)
    gram_martix = torch.mm(feature_map_flatten, feature_map_flatten.t())
    return gram_martix


def compute_content_loss(gen_feature_map, content_feature_map):
    gen_feature_map = torch.squeeze(gen_feature_map)
    con_feature_map = torch.squeeze(content_feature_map)
    loss = torch.mean((gen_feature_map - con_feature_map) ** 2)
    return loss


def compute_layer_style_loss(gen_feature_map, style_feature_map):
    gen_gram = gram(gen_feature_map)
    style_gram = gram(style_feature_map)
    layer_loss = torch.mean((gen_gram - style_gram) ** 2)
    return layer_loss


def compute_style_loss(gen_output, style_output, weight, d, H, W):
    loss = 0
    for i in range(len(weight)):
        loss += weight[i] * compute_layer_style_loss(gen_output["conv" + str(i + 1)],
                                                     style_output["conv" + str(i + 1)]) / (d * H * W)
    return loss


def compute_tv_loss(Y_hat):
    return 0.5 * (torch.abs(Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).mean() +
                  torch.abs(Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).mean())


def image_postprocessing(img, img_size):
    trans_resize = torchvision.transforms.Resize((img_size[1], img_size[0]))
    trans_toPIL = torchvision.transforms.ToPILImage(mode="RGB")
    trans_compose = torchvision.transforms.Compose([trans_resize, trans_toPIL])
    img = torch.squeeze(img)
    img = torch.clamp(img.permute(1, 2, 0) * rgb_std + rgb_mean, 0, 1)
    img = img.permute(2, 0, 1)
    img = trans_compose(img)
    return img


weight = [1, 1e2, 10]
style_weight = [1, 0.8, 0.5, 0.3, 0.1]
extract_list = [0, 5, 10, 19, 28]
style_image_path = "22.jpeg"
content_image_path = "content.png"
style_img, style_img_size = image_preprocessing(style_image_path)
content_img, content_img_size = image_preprocessing(content_image_path)
style_img = style_img.cuda()
content_img = content_img.cuda()
_, d, H, W = content_img.shape
vgg = MyVgg16()
vgg = vgg.cuda()
gen_img = content_img.clone()
gen_img.requires_grad = True
gen_img = gen_img.cuda()
optimizer = torch.optim.Adam([gen_img], lr=0.003)
writter = SummaryWriter("log")
trans_totensor = torchvision.transforms.ToTensor()
writter.add_image("style_image&content_image", trans_totensor(image_postprocessing(style_img, style_img_size)), 0)
writter.add_image("style_image&content_image", trans_totensor(image_postprocessing(content_img, content_img_size)), 1)
for epoch in range(5000):
    optimizer.zero_grad()
    style_output = vgg(style_img, extract_list)
    content_output = vgg(content_img, extract_list)
    gen_output = vgg(gen_img, extract_list)
    content_loss = compute_content_loss(gen_output["conv3"], content_output["conv3"]).cuda()
    style_loss = compute_style_loss(gen_output, style_output, style_weight, d, H, W).cuda()
    tv_loss = compute_tv_loss(gen_img).cuda()
    total_loss = (weight[0] * content_loss + weight[1] * style_loss + weight[2] * tv_loss).cuda()
    total_loss.backward()
    optimizer.step()
    writter.add_scalar("loss", total_loss, epoch)
    if (epoch % 100 == 0):
        print("total_loss: " + str(total_loss))
        print("content_loss: " + str(content_loss))
        print("style_loss: " + str(style_loss))
        print("tv_loss:" + str(tv_loss))
        writter.add_image("result", trans_totensor(image_postprocessing(gen_img, content_img_size)), epoch,
                          dataformats="CHW")
img = image_postprocessing(gen_img, content_img_size)
img.save("output.jpg")
