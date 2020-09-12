# models.py

class ModelOpts:
    L1 = "L1"
    L2 = "L2"
    BN = "BN"
    GBN = "GBN"
    Save = "Save Model"

    def get_name(opts):
      nameParts = []
      for opt in opts:
        if opt == ModelOpts.Save:
          continue
        nameParts.append(opt)
      return "+".join(nameParts)


# Source: https://github.com/apple/ml-cifar-10-faster/blob/master/utils.py#L147
class GhostBatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits
        self.register_buffer('running_mean', torch.zeros(num_features * self.num_splits))
        self.register_buffer('running_var', torch.ones(num_features * self.num_splits))

    def train(self, mode=True):
        if (self.training is True) and (mode is False):  # lazily collate stats when we are going to use them
            self.running_mean = torch.mean(self.running_mean.view(self.num_splits, self.num_features), dim=0).repeat(
                self.num_splits)
            self.running_var = torch.mean(self.running_var.view(self.num_splits, self.num_features), dim=0).repeat(
                self.num_splits)
        return super().train(mode)

    def forward(self, input):
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            return F.batch_norm(
                input.view(-1, C * self.num_splits, H, W), self.running_mean, self.running_var,
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).view(N, C, H, W)
        else:
            return F.batch_norm(
                input, self.running_mean[:self.num_features], self.running_var[:self.num_features],
                self.weight, self.bias, False, self.momentum, self.eps)

class MNIST_S6(nn.Module):
  
    def conv_block (self, in_channels, out_channels, kernel_size, padding = 1):
      return nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, padding = padding, bias = False),
            nn.ReLU(),
            self.bn(out_channels),
            nn.Dropout(0.01))
        
    def out_block(self, in_channels, kernel_size = 1):
      return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=10, kernel_size = kernel_size, padding = 0, bias = False))

    def __init__(self, opts=[]):
        super(MNIST_S6, self).__init__()
        if ModelOpts.GBN in opts:
          self.bn = partial(GhostBatchNorm, num_splits=2)
        else:
          self.bn = nn.BatchNorm2d
        self.conv1 = self.conv_block(1, 10, 3)
        self.conv2 = self.conv_block(10, 10, 3, 0)
        self.conv3 = self.conv_block(10, 11, 3)
        self.conv4 = self.conv_block(11, 11, 3, 0)
        self.conv5 = self.conv_block(11, 12, 3)
        self.conv6 = self.conv_block(12, 12, 3, 0)
        self.conv8 = self.out_block(12, 3)
        self.pool = nn.MaxPool2d(2, 2)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv8(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
    
    
    
class CIFAR10_S7(nn.Module):
    
    def conv_block (self, in_channels, out_channels, kernel_size, padding = 1) :
        return nn.Sequential(
              nn.Conv2d (in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, padding = padding, bias = False),
              nn.ReLU(),
              nn.BatchNorm2d(out_channels),
              nn.Dropout(0.1))
    
    def trans_block (self, in_channels, out_channels):
      return nn.Sequential(
              nn.MaxPool2d(2, 2),
              nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size = 1, padding = 0, bias = False))
      
    def dep_sep_block (self, in_channels, out_channels):
      return nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = in_channels, kernel_size = 3, stride=1, padding = 1, bias = False, groups = in_channels),
            nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 1, stride=1, padding = 0, bias = False, groups = 1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.1))
      
    def dilated_block (self, in_channels, out_channels):
      return nn.Sequential(
          nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3, stride=1, padding = 2, bias = False, dilation = 2),
          nn.ReLU(),
          nn.BatchNorm2d(out_channels),
          nn.Dropout(0.1))
      
    def out_block(self, in_channels, kernel_size = 1):
      return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=10, kernel_size = kernel_size, padding = 0, bias = False))

    def __init__(self, opts=[]):
        super(Net, self).__init__()
        self.conv1 = self.conv_block(3, 64, 3)
        self.conv2 = self.conv_block(64, 128, 3)
        self.trans1 = self.trans_block(128, 32)
        self.sep = self.dep_sep_block(32, 64)
        self.conv3 = self.conv_block(64, 128, 3)
        self.trans2 = self.trans_block(128, 32)
        self.dil = self.dilated_block(32, 64)
        self.conv4 = self.conv_block(64, 128, 3)
        self.trans3 = self.trans_block(128, 32)
        self.gap = nn.Sequential(nn.AvgPool2d(kernel_size=4))
        self.out = self.out_block(32, 1)
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.trans1(x)
        x = self.sep(x)
        x = self.conv3(x)
        x = self.trans2(x)
        x = self.dil(x)
        x = self.conv4(x)
        x = self.trans3(x)
        x = self.gap(x)
        x = self.out(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
