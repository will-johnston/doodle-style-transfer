# https://github.com/paulwarkentin/pytorch-neural-doodle/blob/master/src/
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.autograd import Variable
import cv2
import PIL
import matplotlib.pyplot as plt

import numpy as np
import re
import os
import sys
import bz2
import math
import time
import pickle
import argparse
import itertools
import collections
import pdb

import StyleLoss
import ContentLoss

# Mounts your own google drive, so no files are shared between users.
#from google.colab import drive
#drive.mount('/content/gdrive')

class ansi:
    YELLOW = '\033[0;33m'
    RED = '\033[0;31m'
    ENDC = '\033[0m'
    BOLD = '\033[1;97m'

class Generator():
  def __init__(self):
    '''Takes in the model and performs any necessary data loading. 
    Loads in the images and annotations, and runs style transfer
    using a patch based algorithm in multiple phases.
    '''
    self.featureExtractor = FeatureExtractor()
    self.total_variation_loss = ContentLoss.TotalVariationLoss()
#     self.load_model()
#     args = self.read_args()
    self.load_and_verify_inputs()
    plt.ion()

    
  def load_image(self, filename):
      '''Tries to open an image as a tensor. 
         If file doesn't exist or cannot be opened, return None
      '''
      try:
        img = PIL.Image.open(filename).convert('RGB')
      except IOError:
        return None
      trans = transforms.ToTensor()
      return trans(img)

  def load_and_verify_inputs(self):
    '''Attempts to load the input files. Verify that the files match a valid input.
       Adjusts semantic/content weight if certain inputs are omitted and initializes
       parameters as needed.
    '''
    self.generated_image = None  # this will be loaded and updated in train()
    self.content_image_og = None
    self.content_map_og = None
    self.style_image_og = None
    self.style_map_og = None
    global semantic_weight
    global content_weight
    # get content image, content map
    if content is not None:
      self.content_image_og = self.load_image(content)
      self.content_map_og = self.load_image("{}_sem.png".format(os.path.splitext(content)[0]))
    elif output is not None:
      self.content_image_og = self.load_image(output)
      self.content_map_og = self.load_image("{}_sem.png".format(os.path.splitext(output)[0]))
    
      
    # get style image, style map
    self.style_image_og = self.load_image(style)
    self.style_map_og = self.load_image("{}_sem.png".format(os.path.splitext(style)[0]))
    
    # see if there is either content image or content map
    if self.content_image_og is None and self.content_map_og is None:
      print("No content image or content map found. The result is entirely dependent on the seed.")
    
    # make sure there is a style image
    if self.style_image_og is None:
      print("ERR: Could not find or load the style image.")
      sys.exit(-1)
    
    # make sure that if there is a map for one image (style or content), 
    # there is one for the other
    if (self.content_map_og is None and self.style_map_og is not None) or (self.content_map_og is not None and self.style_map_og is None):
      print("ERR: One of the semantic maps is missing. Make sure the images have a corresponding '_sem.png' image")
      sys.exit(-1)
    
    # if no content map, create zeros tensor of shape style image,
    # and set semantic weight to zero
    if self.content_map_og is None:
      if output_size and self.content_image_og is None:
        shape = []
        shape.append(3)
        for size in output_size.split('x'):
          shape.append(int(size))
      else:
        # use the style image shape
        shape = self.style_image_og.shape
      self.content_map_og = torch.zeros(shape[0], shape[1], shape[2], device=device)
      semantic_weight = 0.0
    
    # if style map doesn't exist, create zeros tensor of shape style image,
    # set semantic weight to zero
    if self.style_map_og is None:
      shape = self.style_image_og.shape
      self.style_map_og = torch.zeros(shape[0], shape[1], shape[2], device=device)
      semantic_weight = 0.0
    
    # if content image doesn't exist, create content image of shape content map,
    # and set content weight to 0
    if self.content_image_og is None:
      shape = self.content_map_og.shape
      self.content_image_og = torch.zeros(shape[0], shape[1], shape[2], device=device)
      content_weight = 0.0
      
    # assert content map and style map have same number of channels
    if self.content_map_og.shape[0] != self.style_map_og.shape[0]:
      print("ERR: Content map and style map have different channel sizes.")
      sys.exit(-1)
      
    if semantic_weight != 0.0:
      semantic_weight = math.sqrt(9.0 / semantic_weight)
    
  def load_generated_image(self, shape=None, scale=None):
    '''Load the generated image. If the seed is noise, started with an image of noise.
       If it is content, start with a copy of the content image. This is called during training.
    '''
    if self.generated_image is None:
      if seed == 'noise':
        # create random tensor of correct shape within specified range
        self.generated_image = torch.randn(shape).to(device).requires_grad_()
      if seed == 'content':
        # create copy of the content image. The content image will already be at the proper
        # scale for the current phase.
        self.generated_image = self.content_image.data.clone()
        # un-normalize the image, to make the generated image start as the actual content img
#         self.generated_image = self.unnormalize(self.generated_image).to(device)
        self.generated_image.requires_grad = True
    


    else:
      # A phase has already been completed in train() if here, so interpolate image to new size.
      self.generated_image = F.interpolate(self.generated_image.detach(),
                                           size=[shape[2], shape[3]], mode='bilinear', align_corners=True)
      self.generated_image.requires_grad = True

#     self.show_image(self.generated_image)
      
  def unnormalize(self, image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    for t, m, s in zip(image, mean, std):
          t.mul_(s).add_(m)
    return image
  
  def resize_image(self, img, size=None, scale=None):
      if size is not None:
          img = img.resize((size, size), Image.ANTIALIAS)
      elif scale is not None:
          img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
      return img


  def save_image(self, filename, data, param1="smoothness", param2="semantic_weight"):
    if param1 == "smoothness":
      filename = "{}{}{}_{}_{}.png".format(filename, alternate_foldername, "output", content_weight, style_weight)
    else:
      filename = "{}{}{}_{}_{}.png".format(filename, output_foldername, "output", param1, param2)
      
    print(filename)
    image = data.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1,2,0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    plt.imsave(filename, image)
    
  def show_image(self, tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1,2,0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    plt.imshow(image)
    plt.pause(0.001)
    
  def prepare_images(self, scale):
    '''At the beginning of a phase, resize the images before running style transfer.
      Then normalize the style and content image, and store the
      3x3 style patches for each style output layer.
    '''
    content_size = [int(x * scale) for x in self.content_image_og.shape[1:]]
    style_size = [int(x * scale) for x in self.style_image_og.shape[1:]]
    transform_content = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(content_size),
        transforms.CenterCrop(content_size),
        transforms.ToTensor(),
       transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                             std=(0.229, 0.224, 0.225))])
    transform_style = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(style_size),
        transforms.CenterCrop(content_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                             std=(0.229, 0.224, 0.225))])
#     self.show_image(self.content_image_og)

    self.content_image = transform_content(self.content_image_og.cpu().clone())[:3,:,:].unsqueeze(0).to(device)
    self.content_map = transform_content(self.content_map_og.cpu().clone())[:3,:,:].unsqueeze(0).to(device)
    self.style_image = transform_style(self.style_image_og.cpu().clone())[:3,:,:].unsqueeze(0).to(device)
    self.style_map = transform_style(self.style_map_og.cpu().clone())[:3,:,:].unsqueeze(0).to(device)
#     self.show_image(self.content_image)
#     self.show_image(self.content_map)
#     self.show_image(self.style_map)

#     self.show_image(self.style_image)
    

    
    
  def prepare_loss(self):
    # run a forward pass on the content image to get a content target
    # this will be used to calculate the content loss 
    _, self.content_target = self.featureExtractor(self.content_image)
    # run a forward pass on the style image and style map
    self.style_target, _ = self.featureExtractor(self.style_image)

    # detach all the outputs
    self.content_target = [(x.detach()) for x in self.content_target]
    self.style_target = [(x.detach()) for x in self.style_target]
    self.content_loss = ContentLoss.ContentLoss(self.content_target)
    self.style_loss = StyleLoss.StyleLoss(self.style_target, self.style_map, self.content_map)
    
    
  def read_args(self):
    parser = argparse.ArgumentParser(description="argument parser for neural style transfer")
    
    parser.add_argument("--style-image", type=str, default="images/style-images/mosaic.jpg",
                                  help="path to style-image")
    parser.add_argument("--content-image", type=str, default="images/style-images/mosaic.jpg",
                                  help="path to style-image")
    train_arg_parser.add_argument("--image-size", type=int, default=256,
                                  help="size of training images, default is 256 X 256")
    train_arg_parser.add_argument("--style-size", type=int, default=None,
                                  help="size of style-image, default is the original size of style image")
    return parser.parse_args()
    
    
  def train(self):
    '''Trains the generated image to adopt the style of the style image and the 
      content of the content image. Runs in phases of increasing resolutions'''
    
    global smoothness
    print(ansi.BOLD + 'Running style transfer...\n' + ansi.ENDC)
    for phase in range(phases):
      # Get necessary arguments to prepare images for current phase
      exp = phases - phase - 1  # get the exponent for the scaling factor
      scale = 1.0 / (2.0 ** exp)
      
      # Prepare the images for the current phase
      self.prepare_images(scale)
      self.prepare_loss()
      self.load_generated_image(shape=self.content_image.shape, scale=1/scale)
      self.optimizer = optim.LBFGS([self.generated_image])  

      print(ansi.RED + "PHASE {} of {}. Current image resolution: {}x{}"
            .format(phase + 1, phases, self.content_image.shape[2], self.content_image.shape[3]) + ansi.ENDC)      
            
      for i in range(iterations):
        
        # Zero the gradients of the optimizer
        def closure():
          self.optimizer.zero_grad()

          # Get outputs from forward pass of model using 
          # current activations and content semantic map
          style_outputs, content_outputs = self.featureExtractor(self.generated_image)

          # Compute the style and content loss and total variation loss          
          c_loss = self.content_loss.compute_loss(content_outputs)
          s_loss = self.style_loss.compute_loss(style_outputs)
          v_loss = self.total_variation_loss(self.generated_image)
          loss = c_loss * content_weight + s_loss * style_weight + v_loss * smoothness
          # Computed the weighted loss
          loss.backward()

          return loss
          # Update the generated image based on the loss
        loss_value = self.optimizer.step(closure)
        if i % print_every == 0:
          print(ansi.YELLOW + "  After iteration {:d}, Loss: {:0.6f}".format(i, loss_value.item()) + ansi.ENDC)
          self.show_image(self.generated_image)
        if i % save_every == 0:
          self.save_image(output_location, self.generated_image, phase, i)
          
      # increase smoothness weight as phases go on to avoid corrupting the image while in low resolution,
      # but removing noise at later phases once things have settled
      smoothness = smoothness + 30
          
    self.save_image(output_location, self.generated_image)




# Global vars, using defaults
content = './images/Landscape.jpg'
# content = None
content_weight = 60.0
content_layers = ['conv4_2']
style = './images/Renoir.jpg'
style_weight = 40.0
style_layers = ['conv3_1', 'conv4_1']
semantic_ext = '_sem.png'
semantic_weight = 40
output = None
# output = '/content/gdrive/My Drive/neural_doodle_images/Coastline.jpg'
output_location = './images/'
output_size = None
phases = 4
seed = 'noise'
iterations = 140
device = 'gpu'
print_every = 20
save_every = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

alternate_foldername = "renoir/"
try:
  os.mkdir(output_location+alternate_foldername)
except:
  pass


generator = Generator()    
      
smoothness = 10
semantic_weight = 10
content_weight = 100
style_weight =  60
variety = 0.0
output_foldername = "renoiri/"
try:
  os.mkdir(output_location+output_foldername)
except:
  pass
    
generator.train()