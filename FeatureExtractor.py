import torchvision.models as models


class FeatureExtractor():
  def __init__(self, device):
    self.model = (models.vgg19(pretrained=True).features).to(device)
    # freeze all VGG parameters since we're only optimizing the target image
    for param in self.model.parameters():
        param.requires_grad_(False)
    self.style_layers = {
                  '10': 'conv3_1', 
                  '19': 'conv4_1',
                  }
    
    self.content_layers  = {
        '21': 'conv4_2'
    }
  def __call__(self, img):
    style_features = []
    content_features = []
    x = img
    for name, layer in self.model._modules.items():
      x = layer(x)
      if name in self.style_layers:
          style_features.append(x)
          
      if name in self.content_layers:
          content_features.append(x)
          
    return style_features, content_features