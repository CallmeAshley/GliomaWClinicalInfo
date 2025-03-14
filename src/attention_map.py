import os
import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from models.dim2.ViT.ViT_explanation_generator import LRP

 
def minmax_norm(img):
    img = (img - img.min()) / (img.max() - img.min())
    return img


class GenerateViTAttMap(nn.Module):
    def __init__(self):
        super(GenerateViTAttMap, self).__init__()
        
    def plot_and_save(self, net, original_image, seg_slice, name, cfg, class_index=None):
        self.attribution_generator = LRP(net)

        transformer_attribution = self.attribution_generator.generate_LRP(original_image, method="transformer_attribution", index=class_index, start_layer=0).detach()
        transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
        transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=16, mode='bilinear')
        transformer_attribution = transformer_attribution.reshape(224, 224).data.cpu().numpy()
        # transformer_attribution = attribution_generator.generate_LRP(original_image, method="full", index=class_index, start_layer=0).detach()[0].data.cpu().numpy()
        transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())
        
        # Plot T2 Image, B, C, H, W
        all_image = original_image.permute(1, 2, 3, 0).data.cpu().numpy() # C, H, W, 1
        seg_slice = seg_slice.permute(1, 2, 0).data.cpu().numpy()
        seg_slice = np.expand_dims(seg_slice, axis=0)
        all_image = np.concatenate((all_image, seg_slice), axis=0) # C x H x W x 1

        if cfg.data.seq_comb == '4seq':
            fig = plt.figure(frameon=False, dpi=600)
            for i in range(5):
                ax = fig.add_subplot(2,5,i+1)
                ax.imshow(minmax_norm(all_image[i]), cmap=plt.cm.gray, interpolation='nearest')
                ax.axis('off')
            
                ax = fig.add_subplot(2,5,i+6)
                ax.imshow(minmax_norm(all_image[i]), cmap=plt.cm.gray, interpolation='nearest')
                ax.imshow(transformer_attribution, cmap=plt.cm.jet, alpha=.5, interpolation='bilinear')
                ax.axis('off')

        elif cfg.data.seq_comb == 't2':
            fig = plt.figure(frameon=False, dpi=600)
            for i in range(2):
                ax = fig.add_subplot(2,2,i+1)
                ax.imshow(minmax_norm(all_image[i]), cmap=plt.cm.gray, interpolation='nearest')
                ax.axis('off')
            
                ax = fig.add_subplot(2,2,i+2)
                ax.imshow(minmax_norm(all_image[i]), cmap=plt.cm.gray, interpolation='nearest')
                ax.imshow(transformer_attribution, cmap=plt.cm.jet, alpha=.5, interpolation='bilinear')
                ax.axis('off')
        
        plt.tight_layout()
        plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)
        
        save_path = os.path.join(cfg.paths.data_root, 'images', 'AttMap')
        os.makedirs(save_path, exist_ok=True)
        
        plt.savefig(os.path.join(save_path, name + '.png'))
        plt.close()
        plt.clf()



class GenerateCNNGradCAM(nn.Module):
    def __init__(self):
        super(GenerateCNNGradCAM, self).__init__()

    def plot_and_save(self, net, original_image, seg_slice, name, cfg, class_index=None):
        target_layers = [net.layer4[-1]]
        self.attribution_generator = GradCAM(model=net, target_layers=target_layers, use_cuda=True)
        
        targets = [ClassifierOutputTarget(class_index)]
        grayscale_cam = self.attribution_generator(input_tensor=original_image, targets=targets)
        # grayscale_cam = np.expand_dims(grayscale_cam, 3)[0]
        grayscale_cam = grayscale_cam.transpose(1, 2, 0)
        grayscale_cam = (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max() - grayscale_cam.min())


        # Plot T2 Image, B, C, H, W
        all_image = original_image.permute(1, 2, 3, 0).data.cpu().numpy() # C, H, W, 1
        seg_slice = seg_slice.permute(1, 2, 0).data.cpu().numpy()
        seg_slice = np.expand_dims(seg_slice, axis=0)
        all_image = np.concatenate((all_image, seg_slice), axis=0) # C x H x W x 1

        if cfg.data.seq_comb == '4seq':
            fig = plt.figure(frameon=False, dpi=600)
            for i in range(5):
                ax = fig.add_subplot(2,5,i+1)
                ax.imshow(minmax_norm(all_image[i]), cmap=plt.cm.gray, interpolation='nearest')
                ax.axis('off')
            
                ax = fig.add_subplot(2,5,i+6)
                ax.imshow(minmax_norm(all_image[i]), cmap=plt.cm.gray, interpolation='nearest')
                ax.imshow(grayscale_cam, cmap=plt.cm.jet, alpha=.5, interpolation='bilinear')
                ax.axis('off')

        elif cfg.data.seq_comb == 't2':
            fig = plt.figure(frameon=False, dpi=600)
            for i in range(2):
                ax = fig.add_subplot(2,2,i+1)
                ax.imshow(minmax_norm(all_image[i]), cmap=plt.cm.gray, interpolation='nearest')
                ax.axis('off')
            
                ax = fig.add_subplot(2,2,i+2)
                ax.imshow(minmax_norm(all_image[i]), cmap=plt.cm.gray, interpolation='nearest')
                ax.imshow(grayscale_cam, cmap=plt.cm.jet, alpha=.5, interpolation='bilinear')
                ax.axis('off')
            
        plt.tight_layout()
        plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)

        save_path = os.path.join(cfg.paths.data_root, 'images', 'GradCAM')
        os.makedirs(save_path, exist_ok=True)
        
        plt.savefig(os.path.join(save_path, name + '.png'))
        plt.close()
        plt.clf()

 
class GenerateCNNLRP(nn.Module):
    def __init__(self):
        super(GenerateCNNLRP, self).__init__()
        
    def plot_and_save(self, net, original_image, seg_slice, name, cfg, class_index=None):
        outputs = net(original_image)

        R = torch.ones(outputs.shape).cuda()
        R = net.fc.relprop(R, 1)
        R = R.reshape_as(net.avgpool.Y)
        R4 = net.avgpool.relprop(R, 1)
        R3 = net.layer4.relprop(R4, 1)
        R2 = net.layer3.relprop(R3, 1)
        R1 = net.layer2.relprop(R2, 1)
        R0 = net.layer1.relprop(R1, 1)
        R0 = net.maxpool.relprop(R0, 1)
        R0 = net.relu.relprop(R0, 1)
        R0 = net.bn1.relprop(R0, 1)
        R_final = net.conv1.relprop(R0, 1)
        R_final = R_final[0]
            
        R_t1, R_t1c, R_t2, R_flair = R_final
    
        
        R_t1 = minmax_norm(R_t1)
        R_t1c = minmax_norm(R_t1c)
        R_t2 = minmax_norm(R_t2)
        R_flair = minmax_norm(R_flair)
        zero_array = torch.zeros_like(R_flair).cuda()
        
        R_final = torch.stack([R_t1, R_t1c, R_t2, R_flair, zero_array], 0)
        R_final = R_final.detach().cpu().numpy()
        
        # Plot T2 Image, B, C, H, W
        all_image = original_image.permute(1, 2, 3, 0).data.cpu().numpy() # C, H, W, 1
        seg_slice = seg_slice.permute(1, 2, 0).data.cpu().numpy()
        seg_slice = np.expand_dims(seg_slice, axis=0)
        all_image = np.concatenate((all_image, seg_slice), axis=0) # C x H x W x 1

        if cfg.data.seq_comb == '4seq':
            fig = plt.figure(frameon=False, dpi=600)
            for i in range(5):
                ax = fig.add_subplot(2,5,i+1)
                ax.imshow(minmax_norm(all_image[i]), cmap=plt.cm.gray, interpolation='nearest')
                ax.axis('off')
            
                ax = fig.add_subplot(2,5,i+6)
                ax.imshow(minmax_norm(all_image[i]), cmap=plt.cm.gray, interpolation='nearest')
                ax.imshow(R_final[i], cmap=plt.cm.jet, alpha=.5, interpolation='bilinear')
                ax.axis('off')

        elif cfg.data.seq_comb == 't2':
            fig = plt.figure(frameon=False, dpi=600)
            for i in range(2):
                ax = fig.add_subplot(2,2,i+1)
                ax.imshow(minmax_norm(all_image[i]), cmap=plt.cm.gray, interpolation='nearest')
                ax.axis('off')
            
                ax = fig.add_subplot(2,2,i+2)
                ax.imshow(minmax_norm(all_image[i]), cmap=plt.cm.gray, interpolation='nearest')
                ax.imshow(R_final[i], cmap=plt.cm.jet, alpha=.5, interpolation='bilinear')
                ax.axis('off')
        
        plt.tight_layout()
        plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)

        save_path = os.path.join(cfg.paths.data_root, 'images', 'LRP')
        os.makedirs(save_path, exist_ok=True)
        
        plt.savefig(os.path.join(save_path, name + '.png'))
        plt.close()
        plt.clf()        
    