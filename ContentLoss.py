import torch.nn as nn

class ContentLoss(nn.Module):
  def __init__(self, content_outputs):
    super(ContentLoss, self).__init__()
    self.targets = [x.detach() for x in content_outputs]
    self.mse = nn.MSELoss()
    
  def compute_loss(self, outputs):
    self.loss = 0.0
    
    # compare mse loss for each layer of intermediate outputs
    if (len(outputs) == len(self.targets)):
      for layer in range(len(outputs)):
        self.loss += nn.functional.mse_loss(outputs[layer], self.targets[layer])
        
    # weight computed loss and return
    return self.loss

class TotalVariationLoss():
  """
  Total variation denoising, which will be used as part of the loss function.
  Removes noise while preserving important details, such as edges
  """
  def __call__(self, x):  
    loss = (((x[:,:,:-1,:-1] - x[:,:,1:,:-1])**2 + (x[:,:,:-1,:-1] - x[:,:,:-1,1:])**2)**1.25).mean()
    return loss