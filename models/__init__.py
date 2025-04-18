import torchvision
from .add_gcn import ADD_GCN, ImprovedADD_GCN_CBAM

model_dict = {'ADD_GCN': ADD_GCN, "ADD_GCN_CBAM": ImprovedADD_GCN_CBAM}

def get_model(num_classes, args):
    res101 = torchvision.models.resnet101(pretrained=True)
    model = model_dict[args.model_name](res101, num_classes)
    return model