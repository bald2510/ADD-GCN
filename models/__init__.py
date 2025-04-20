import torchvision
from .add_gcn import ADD_GCN, ImprovedADD_GCN_CBAM

# Define a dictionary of models
# Include the ADD_GCN and ImprovedADD_GCN_CBAM models
model_dict = {'ADD_GCN': ADD_GCN, "ADD_GCN_CBAM": ImprovedADD_GCN_CBAM}

def get_model(num_classes, args):
    """
    Get the model based on the model name and number of classes.
    """
    
    res101 = torchvision.models.resnet101(pretrained=True) # Load a pre-trained ResNet-101 model
    model = model_dict[args.model_name](res101, num_classes) # Initialize the model with the ResNet-101 backbone and the specified number of classes
    return model