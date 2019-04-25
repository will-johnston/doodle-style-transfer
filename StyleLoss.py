class StyleLoss(nn.Module):
  def __init__(self, style_outputs, style_map, content_map, patch_size=(3,3), stride=3):
    super(StyleLoss, self).__init__()
    '''Convert the style image into patches, in order to compute the normalized
       cross correlation. 
       This requires downsampling the semantic maps and appending
       them to the model outputs.
    '''
    self.mse = nn.MSELoss()
    self.patched_layers, self.norms = [], []
    style_maps_ds, self.content_maps_ds = [], []  # content maps used in compute_loss
     
    # detach style outputs
    style_outputs = [x.detach() for x in style_outputs]
    style_map = style_map.detach()
    content_map = content_map.detach()
    # Extract the patches from the style image of the current phase
    self.unfold = nn.Unfold(patch_size, stride=stride)
    for layer in range(len(style_outputs)):
      # downsample the style and content map for this layer
      style_map_ds = self.downsample_sem_map(style_map, style_layers[layer])
      style_maps_ds.append(style_map_ds)
      self.content_maps_ds.append(self.downsample_sem_map(content_map, style_layers[layer]))
      
      # weigh the style map
      map_channel_no = style_map_ds.size()[0]
      style_map_weighted = style_map_ds * semantic_weight * map_channel_no
      
      # concatenate the style img output layer with the weighted, downsampled style mapping
      semantic_layer = torch.cat((style_outputs[layer], style_map_weighted), dim=1)
      
      # unfold the semantic layer, get the normalizing factor by computing the L2 norm
      patches = self.unfold(semantic_layer).squeeze(0)
      patch_num = patches.size()[1]
      norm = (1.0/torch.norm(patches, p=2, dim=0)).expand(patch_num, patch_num)
      self.patched_layers.append(patches)
      self.norms.append(norm)
      
      
  def compute_loss(self, gen_outputs):
    '''Compute the style loss from a given model output of the generated image. 
      Here, we will find the normalized cross correlation between the (k x k) patches.
      Then, compute the euclidean distance of the generated image patches' nearest neighbors to the style image patches.
      This is performed for each layer of the generated image model output.
    '''
    loss = 0.0
    for layer in range(len(gen_outputs)):

      # get the style image patches
      style_patches = self.patched_layers[layer]
      patch_num = style_patches.size()[1]
      
      # concatenate output layer with content map
      gen_semantic = torch.cat((gen_outputs[layer], self.content_maps_ds[layer]), dim=1)
      gen_patches = self.unfold(gen_semantic).squeeze(0)
      gen_norm = (1.0/torch.norm(gen_patches, p=2, dim=0)).expand(patch_num, patch_num)
      gen_patches_t = torch.t(gen_patches)
      
      # create cross correlation matrix between generated patches and style image patches.
      # get the dot product between the generated patches and all possible style image patches
      corr = torch.matmul(gen_patches_t, style_patches)  
      # normalize the cross corr matrix
      corr = torch.t(gen_norm) * (self.norms[layer] * corr)  
      # retreive the style image patches which have the highest correlation between the generated patches
      if np.random.random(1) < variety:
        if corr.shape[1] < 4:
          topn = corr.shape[1]
        else:
          topn = 4
          
        nearest_neighs = torch.topk(corr, topn, dim=1)
        nth = np.random.randint(0, topn)
        nearest_neighs = nearest_neighs[1][:,nth]
      else:
        nearest_neighs = torch.argmax(corr, dim=1)
      
      # compute the L2 norm between each generated patch and it's nearest neighbor
      layer_loss = self.mse(gen_patches, style_patches[:, nearest_neighs])
      loss += layer_loss
    return loss
      
  def downsample_sem_map(self, sem_map, layer):
    '''Downsample the semantic map in order to concatenate to a particular VGG layer
    '''
    # Get the first number of the output layer (e.g. relu1_2)
    exp = int(layer[re.search('\d', layer).start()]) - 1      
    # Downsample using nearest neighbors, scale depends on number of pooling layers
    # the net has gone through
    sem_downsampled = F.interpolate(sem_map, scale_factor=1/(2**exp), mode='nearest')
    return sem_downsampled