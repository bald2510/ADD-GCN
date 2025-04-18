import os, sys, pdb
import argparse
from models import get_model
from data import make_data_loader
import warnings
from trainer import Trainer
import torch
import torch.backends.cudnn as cudnn
import random

# Create an argument parser for command-line options
parser = argparse.ArgumentParser(description='PyTorch Training for Multi-label Image Classification')

''' Fixed in general '''
# General settings for data, image size, epochs, batch size, etc.
parser.add_argument('--data_root_dir', default='./datasets/', type=str, help='save path')  # Root directory for datasets
parser.add_argument('--image-size', '-i', default=448, type=int)  # Image size
parser.add_argument('--epochs', default=10, type=int)  # Number of training epochs
parser.add_argument('--epoch_step', default=[30, 40], type=int, nargs='+', help='number of epochs to change learning rate')  # Epochs to adjust learning rate
parser.add_argument('-b', '--batch-size', default=16, type=int)  # Batch size
parser.add_argument('-j', '--num_workers', default=4, type=int, metavar='INT', help='number of data loading workers (default: 4)')  # Number of data loading workers
parser.add_argument('--display_interval', default=200, type=int, metavar='M', help='display_interval')  # Interval for displaying logs
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float)  # Learning rate
parser.add_argument('--lrp', '--learning-rate-pretrained', default=0.1, type=float, metavar='LRP', help='learning rate for pre-trained layers')  # Learning rate for pre-trained layers
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')  # Momentum for optimizer
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')  # Weight decay
parser.add_argument('--max_clip_grad_norm', default=10.0, type=float, metavar='M', help='max_clip_grad_norm')  # Max gradient clipping norm
parser.add_argument('--seed', default=1, type=int, help='seed for initializing training. ')  # Random seed for reproducibility

''' Train setting '''
# Training-specific settings
parser.add_argument('--data', metavar='NAME', help='dataset name (e.g. COCO2014')  # Dataset name
parser.add_argument('--model_name', type=str, default='ADD_GCN')  # Model name
parser.add_argument('--save_dir', default='./checkpoint/COCO2014/', type=str, help='save path')  # Directory to save checkpoints

''' Val or Test setting '''
# Validation or testing-specific settings
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')  # Flag for evaluation mode
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')  # Path to resume from a checkpoint

# Main function to handle training or evaluation
def main(args):
    # Set random seed for reproducibility
    if args.seed is not None:
        print ('* absolute seed: {}'.format(args.seed))
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    # Determine if the mode is training or evaluation
    is_train = True if not args.evaluate else False

    # Create data loaders for training and validation
    train_loader, val_loader, num_classes = make_data_loader(args, is_train=is_train)

    # Initialize the model
    model = get_model(num_classes, args)

    # Define the loss function
    criterion = torch.nn.MultiLabelSoftMarginLoss()

    # Initialize the trainer with model, loss, data loaders, and arguments
    trainer = Trainer(model, criterion, train_loader, val_loader, args)
    
    # Train or validate the model based on the mode
    if is_train:
        trainer.train()
    else:
        trainer.validate()

# Entry point of the script
if __name__ == "__main__":
    args = parser.parse_args()  # Parse command-line arguments
    main(args)  # Call the main function with parsed arguments
