import os
import torch
import torchvision
import pandas as pd
from time import time, strftime, gmtime
from utility import load_data, save_image, create_gif, combined_result


def extract_features(x, cnn, device='cuda'):
    if torch.cuda.is_available():
        x = x.to(device)
    features = []
    prev_feat = x
    for module in cnn._modules.values():
        next_feat = module(prev_feat)
        features.append(next_feat)
        prev_feat = next_feat
    return features


def content_loss(content_weight, content_current, content_original):
    channels = content_current.size(1)
    content_current = content_current.view(channels, -1)
    channels = content_original.size(1)
    content_original = content_original.view(channels, -1)
    loss = content_weight * ((content_current - content_original) ** 2).sum()
    return loss


def style_loss(feats, style_layers, style_targets, style_weights):
    losses = []
    for i, layer in enumerate(style_layers):
        features = feats[layer]
        N, C, H, W = features.shape
        features = features.view(N, C, -1)
        gram = torch.bmm(features, features.permute([0, 2, 1]))  # gram matrix
        gram = gram / (C * H * W)
        losses.append(style_weights[i] * ((gram - style_targets[i]) ** 2).sum())
    style_loss = sum(losses)
    return style_loss


def tv_loss(img, tv_weight):
    N, C, H, W = img.size()
    h_var = img[:, :, torch.arange(1, H).long(), :] - img[:, :, torch.arange(0, H - 1).long(), :]
    w_var = img[:, :, :, torch.arange(1, W).long()] - img[:, :, :, torch.arange(0, W - 1).long()]
    h_var, w_var = (h_var ** 2).sum(), (w_var ** 2).sum()
    return tv_weight * (h_var + w_var)


def train(cnn=None,
          content_image=None,
          style_image=None,
          content_layer=0,
          content_weight=0.,
          style_layers=[],
          style_weights=[],
          tv_weight=0.,
          init_random=False,
          num_epochs=0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Extract features of content image
    content_img = load_data(input_img=content_image, size=512)
    feats = extract_features(content_img, cnn, device=device)
    content_target = feats[content_layer].clone()

    # Extract features of style image
    style_img = load_data(input_img=style_image, size=512)
    feats = extract_features(style_img, cnn, device=device)
    style_targets = []
    for idx in style_layers:
        features = feats[idx].clone()
        N, C, H, W = features.shape
        features = features.view(N, C, -1)
        gram = torch.bmm(features, features.permute([0, 2, 1]))  # gram matrix
        gram = gram / (C * H * W)
        style_targets.append(gram)

    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    # Initialize output image to content image or nois
    if init_random:
        img = torch.Tensor(content_img.size()).uniform_(0, 1).type(dtype)
    else:
        img = content_img.clone().type(dtype)

    # We do want the gradient computed on our image!
    img.requires_grad_()
    if torch.cuda.is_available():
        img = img.to(device)

    lr = 3.0

    optimizer = torch.optim.Adam([img], lr=lr)

    epoch_list, content_loss_list, style_loss_list = [], [], []  # for evaluation purpose

    for epoch in range(num_epochs):
        img.data.clamp_(-1.5, 1.5)

        optimizer.zero_grad()

        feats = extract_features(img, cnn, device=device)

        # Compute loss
        c_loss = content_loss(content_weight, feats[content_layer], content_target)
        s_loss = style_loss(feats, style_layers, style_targets, style_weights)
        t_loss = tv_loss(img, tv_weight)
        loss = c_loss + s_loss + t_loss

        loss.backward()

        # Perform gradient descents on our image values

        if epoch == 180:
            optimizer = torch.optim.Adam([img], lr=0.1)

        optimizer.step()

        '''
        # for evaluation purpose
        epoch_list.append(epoch)
        content_loss_list.append('{:.6f}'.format(c_loss))
        style_loss_list.append('{:.6f}'.format(s_loss))
        '''

        if (epoch % 4 == 0 and epoch > 0) or (epoch + 1) == num_epochs:
            print('epoch = {} | content_loss = {:.6f} | style_loss = {:.6f}'.format(epoch, c_loss, s_loss))
            save_image(img=img, name="result/output/output_{}.png".format(epoch))

    combined_result(img=img, content_img_path='data/content.jpg', style_img_path='data/style.jpg')
    if len(list(os.listdir('result/output'))) is not 0:
        create_gif(path='result/output')

    '''
    # for evaluation purpose
    df = pd.DataFrame({'epoch': epoch_list, 'content_loss': content_loss_list, 'style_loss': style_loss_list})
    df.to_csv('log_{}.csv'.format(num_epochs))
    '''


def main():
    content_image_path = 'data/content.jpg'
    style_image_path = 'data/style.jpg'
    if not os.path.exists('result/output'):
        os.makedirs('result/output')
    print('Training on {}'.format(torch.cuda.get_device_name(0)))
    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    cnn = torchvision.models.squeezenet1_1(pretrained=True).features
    cnn.type(dtype)
    # disable autograd
    for param in cnn.parameters():
        param.requires_grad = False

    start = time()

    train(cnn=cnn,
          content_image=content_image_path,
          style_image=style_image_path,
          content_layer=3,
          content_weight=0.02,
          style_layers=[1, 4, 6, 7],
          style_weights=[300000, 1500, 15, 3],
          tv_weight=0.2,
          init_random=False,
          num_epochs=240)

    end = time()
    print('Training Done | Time Elapsed --> {} seconds'.format(strftime('%H:%M:%S', gmtime(end - start))))


if __name__ == '__main__':
    main()
