import torch
import torch.nn as nn


class DynamicGraphConvolution(nn.Module):
    '''
    Dynamic Graph Convolutional Network (D-GCN) module, used to learn the dynamic graph representations of the input data.
    The module consists of two main components:
    1. Static Graph Convolution: This part computes the static graph representations of the input data.
    2. Dynamic Graph Convolution: This part computes the dynamic graph representations of the input data.
    
    The static graph representations are computed using a static adjacency matrix, while the dynamic graph representations are computed using a dynamic adjacency matrix.
    The dynamic adjacency matrix is constructed using the global representations of the input data.
    The dynamic graph representations are then used to compute the final output of the module.
    
    The module also includes a residual connection to combine the static and dynamic graph representations.
    The output of the module is the final dynamic graph representations.
    
    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        num_nodes (int): Number of nodes in the graph.
    '''
    def __init__(self, in_features, out_features, num_nodes):
        '''
        Initialize the D-GCN module.
        
        Architecture:
        1. Static Graph Convolution: This part computes the static graph representations of the input data.
        2. Dynamic Graph Convolution: This part computes the dynamic graph representations of the input data.
        3. Residual Connection: This part combines the static and dynamic graph representations.
        4. Output: The output of the module is the final dynamic graph representations.
        
        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            num_nodes (int): Number of nodes in the graph.
        '''
        
        super(DynamicGraphConvolution, self).__init__()

        self.static_adj = nn.Sequential(
            nn.Conv1d(num_nodes, num_nodes, 1, bias=False),
            nn.LeakyReLU(0.2))
        self.static_weight = nn.Sequential(
            nn.Conv1d(in_features, out_features, 1),
            nn.LeakyReLU(0.2))

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.conv_global = nn.Conv1d(in_features, in_features, 1)
        self.bn_global = nn.BatchNorm1d(in_features)
        self.relu = nn.LeakyReLU(0.2)

        self.conv_create_co_mat = nn.Conv1d(in_features*2, num_nodes, 1)
        self.dynamic_weight = nn.Conv1d(in_features, out_features, 1)

    def forward_static_gcn(self, x):
        '''
        Static GCN module
        Use the static adjacency matrix to compute the static graph representations of the input data.
        The static adjacency matrix is computed using a static graph convolutional layer.
        The static graph representations are then used to compute the final output of the module.
        The output of the module is the final static graph representations.
        '''
        
        x = self.static_adj(x.transpose(1, 2))
        x = self.static_weight(x.transpose(1, 2))
        return x

    def forward_construct_dynamic_graph(self, x):
        '''
        Construct the dynamic graph representations of the input data.
        The dynamic graph representations are computed using a dynamic adjacency matrix.
        The dynamic adjacency matrix is constructed using the global representations of the input data.
        The dynamic graph representations are then used to compute the final output of the module.
        The output of the module is the final dynamic graph representations.
        '''
        
        ### Model global representations ###
        x_glb = self.gap(x)
        x_glb = self.conv_global(x_glb)
        x_glb = self.bn_global(x_glb)
        x_glb = self.relu(x_glb)
        x_glb = x_glb.expand(x_glb.size(0), x_glb.size(1), x.size(2))
        
        ### Construct the dynamic correlation matrix ###
        x = torch.cat((x_glb, x), dim=1)
        dynamic_adj = self.conv_create_co_mat(x)
        dynamic_adj = torch.sigmoid(dynamic_adj)
        return dynamic_adj

    def forward_dynamic_gcn(self, x, dynamic_adj):
        '''
        Dynamic GCN module
        Use the dynamic adjacency matrix to compute the dynamic graph representations of the input data.
        The dynamic adjacency matrix is constructed using the global representations of the input data.
        The dynamic graph representations are then used to compute the final output of the module.
        The output of the module is the final dynamic graph representations.
        '''
        x = torch.matmul(x, dynamic_adj)
        x = self.relu(x)
        x = self.dynamic_weight(x)
        x = self.relu(x)
        return x

    def forward(self, x):
        """ D-GCN module

        Shape: 
        - Input: (B, C_in, N) # C_in: 1024, N: num_classes
        - Output: (B, C_out, N) # C_out: 1024, N: num_classes
        """
        out_static = self.forward_static_gcn(x)
        x = x + out_static # residual
        dynamic_adj = self.forward_construct_dynamic_graph(x)
        x = self.forward_dynamic_gcn(x, dynamic_adj)
        return x

class ChannelAttention(nn.Module):
    """
    Channel Attention module
    This module computes the channel attention weights for the input feature map.
    The channel attention weights are computed using the average and max pooling operations.
    The average and max pooling operations are used to compute the channel-wise statistics of the input feature map.
    The channel attention weights are then computed using a feedforward neural network and used to weight the input feature map.
    The output of the module is the weighted input feature map.
    
    Args:
        in_planes (int): Number of input channels.
        ratio (int): Reduction ratio for the channel attention weights.
    """
    
    def __init__(self, in_planes, ratio=16):
        """
        Initialize the Channel Attention module.
        
        Architecture:
        1. Average Pooling: This part computes the average pooling of the input feature map.
        2. Max Pooling: This part computes the max pooling of the input feature map.
        3. Feedforward Neural Network: This part computes the channel attention weights using a feedforward neural network.
        4. Sigmoid Activation: This part applies the sigmoid activation function to the channel attention weights.
        5. Output: The output of the module is the weighted input feature map.
        """
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Channel Attention module forward pass.
        """
        
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x

class SpatialAttention(nn.Module):
    """
    Spatial Attention module
    This module computes the spatial attention weights for the input feature map.
    The spatial attention weights are computed using the average and max pooling operations.
    The average and max pooling operations are used to compute the spatial-wise statistics of the input feature map.
    The spatial attention weights are then computed using a convolutional layer and used to weight the input feature map.
    The output of the module is the weighted input feature map.
    
    
    Args:
        kernel_size (int): Kernel size for the convolutional layer.
    """
    
    def __init__(self, kernel_size=7):
        """
        Initialize the Spatial Attention module.
        """
        super(SpatialAttention, self).__init__()

        padding = 3

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False) # Convolutional layer
        self.sigmoid = nn.Sigmoid() # Sigmoid activation function

    def forward(self, x):
        # Feedforward pass for the Spatial Attention module
        avg_out = torch.mean(x, dim=1, keepdim=True) # Average pooling
        max_out, _ = torch.max(x, dim=1, keepdim=True) # Max pooling
        out = torch.cat([avg_out, max_out], dim=1) # Concatenate the average and max pooling results
        out = self.conv1(out)
        return self.sigmoid(out) * x # Apply the spatial attention weights to the input feature map

class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM)
    This module combines the channel and spatial attention modules to compute the attention weights for the input feature map.
    The channel and spatial attention weights are computed separately and then combined to weight the input feature map.
    The output of the module is the weighted input feature map.
    """
    
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        """
        CBAM module forward pass.        
        """
        x = self.ca(x) # Apply channel attention
        x = self.sa(x) # Apply spatial attention
        return x

class ADD_GCN(nn.Module):
    """
    ADD_GCN module
    This module combines the ADD and GCN modules to compute the attention weights for the input feature map.
    The ADD and GCN modules are used to compute the attention weights for the input feature map.
    The attention weights are then used to weight the input feature map.
    The output of the module is the weighted input feature map.
    
    """
    def __init__(self, model, num_classes):
        super(ADD_GCN, self).__init__()
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
        self.num_classes = num_classes

        self.fc = nn.Conv2d(model.fc.in_features, num_classes, (1,1), bias=False)

        self.conv_transform = nn.Conv2d(2048, 1024, (1,1))
        self.relu = nn.LeakyReLU(0.2)

        self.gcn = DynamicGraphConvolution(1024, 1024, num_classes)
        
        self.mask_mat = nn.Parameter(torch.eye(self.num_classes).float())
        self.last_linear = nn.Conv1d(1024, self.num_classes, 1)

        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward_feature(self, x):
        x = self.features(x)
        return x

    def forward_classification_sm(self, x):
        """ Get another confident scores {s_m}.

        Shape:
        - Input: (B, C_in, H, W) # C_in: 2048
        - Output: (B, C_out) # C_out: num_classes
        """
        x = self.fc(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = x.topk(1, dim=-1)[0].mean(dim=-1)
        return x

    def forward_sam(self, x):
        """ SAM module

        Shape: 
        - Input: (B, C_in, H, W) # C_in: 2048
        - Output: (B, C_out, N) # C_out: 1024, N: num_classes
        """
        mask = self.fc(x) 
        mask = mask.view(mask.size(0), mask.size(1), -1) 
        mask = torch.sigmoid(mask)
        mask = mask.transpose(1, 2)

        x = self.conv_transform(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = torch.matmul(x, mask)
        return x

    def forward_dgcn(self, x):
        x = self.gcn(x)
        return x

    def forward(self, x):
        x = self.forward_feature(x)

        out1 = self.forward_classification_sm(x)

        v = self.forward_sam(x) # B*1024*num_classes
        z = self.forward_dgcn(v)
        z = v + z

        out2 = self.last_linear(z) # B*1*num_classes
        mask_mat = self.mask_mat.detach()
        out2 = (out2 * mask_mat).sum(-1)
        return out1, out2

    def get_config_optim(self, lr, lrp):
        small_lr_layers = list(map(id, self.features.parameters()))
        large_lr_layers = filter(lambda p:id(p) not in small_lr_layers, self.parameters())
        return [
                {'params': self.features.parameters(), 'lr': lr * lrp},
                {'params': large_lr_layers, 'lr': lr},
                ]


class ImprovedADD_GCN_CBAM(nn.Module):
    """
    Improved ADD_GCN with CBAM module
    This module combines the ADD and GCN modules with the CBAM module to compute the attention weights for the input feature map.
    The ADD and GCN modules are used to compute the attention weights for the input feature map.
    The attention weights are then used to weight the input feature map.
    The output of the module is the weighted input feature map.
    
    Args:
        model (nn.Module): Backbone model (e.g., ResNet).
        num_classes (int): Number of classes for classification.
    """
    def __init__(self, model, num_classes):
        super(ImprovedADD_GCN_CBAM, self).__init__()
        self.features = nn.Sequential( 
            model.conv1, # Converting the input image to feature map
            model.bn1, # Batch normalization
            model.relu, # ReLU activation function
            model.maxpool, # Max pooling layer
            model.layer1, # First layer of the ResNet model
            model.layer2,
            model.layer3,
            model.layer4,
        )
        self.num_classes = num_classes

        self.fc = nn.Conv2d(model.fc.in_features, num_classes, (1,1), bias=False) # Convolutional layer for classification
        
        # convolutional layer for transforming the feature map
        self.conv_transform = nn.Conv2d(2048, 1024, (1,1))
        self.relu = nn.LeakyReLU(0.2) 
        
        # CBAM module for attention mechanism
        self.cbam = CBAM(1024)

        # Dynamic Graph Convolutional Network (D-GCN) module for learning the dynamic graph representations of the input data
        self.gcn = DynamicGraphConvolution(1024, 1024, num_classes)
        
        # Mask matrix for the output of the D-GCN module
        # The mask matrix is used to weight the output of the D-GCN module
        self.mask_mat = nn.Parameter(torch.eye(self.num_classes).float())
        self.last_linear = nn.Conv1d(1024, self.num_classes, 1)

        # image normalization
        # The image normalization parameters are used to normalize the input image
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward_feature(self, x):
        x = self.features(x)
        return x

    def forward_classification_sm(self, x):
        """ Get another confident scores {s_m}.

        Shape:
        - Input: (B, C_in, H, W) # C_in: 2048
        - Output: (B, C_out) # C_out: num_classes
        """
        x = self.fc(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = x.topk(1, dim=-1)[0].mean(dim=-1)
        return x

    def forward_cbam(self, x):
        # CBAM module forward pass.
        x = self.conv_transform(x)     # B x 1024 x H x W
        x = self.cbam(x)               # Áp dụng CBAM
        x = x.view(x.size(0), x.size(1), -1)  # Flatten không gian còn lại -> (B, 1024, H*W)
        
        x = x.topk(self.num_classes, dim=2)[0]  # (B, 1024, num_classes)
        
        return x

    def forward_dgcn(self, x):
        # D-GCN module forward pass.
        x = self.gcn(x)
        return x

    def forward(self, x):
        ''' 
        Forward pass through the model.
        
        The forward pass consists of the following steps:
        1. Feature extraction using the backbone model.
        2. Classification using the fully connected layer.
        3. Attention mechanism using the CBAM module.
        4. Dynamic graph convolution using the D-GCN module.
        5. Output the final classification scores.
        '''
        x = self.forward_feature(x)

        out1 = self.forward_classification_sm(x)

        v = self.forward_cbam(x) # B*1024*num_classes
        z = self.forward_dgcn(v)
        z = v + z

        out2 = self.last_linear(z) # B*1*num_classes
        mask_mat = self.mask_mat.detach()
        out2 = (out2 * mask_mat).sum(-1)
        return out1, out2

    def get_config_optim(self, lr, lrp):
        """ Get the configuration for the optimizer.
        
        The configuration consists of the following parameters:
        1. Learning rate for the backbone model.
        2. Learning rate for the other layers.
        3. Learning rate for the classification layer.s
        4. Learning rate for the D-GCN module.
        """
        small_lr_layers = list(map(id, self.features.parameters()))
        large_lr_layers = filter(lambda p:id(p) not in small_lr_layers, self.parameters())
        return [
                {'params': self.features.parameters(), 'lr': lr * lrp},
                {'params': large_lr_layers, 'lr': lr},
                ]
