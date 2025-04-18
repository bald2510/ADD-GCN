import os, sys, pdb
import math
import torch
from PIL import Image
import numpy as np
import random

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        """Resets all metrics to zero"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Updates the metrics with a new value"""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def average(self):
        """Returns the average value"""
        return self.avg
    
    def value(self):
        """Returns the current value"""
        return self.val

    def __str__(self):
        """String representation of the metrics"""
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class AveragePrecisionMeter(object):
    """
    The APMeter measures the average precision per class.
    Designed to operate on `NxK` Tensors `output` and `target`.
    """

    def __init__(self, difficult_examples=True):
        super(AveragePrecisionMeter, self).__init__()
        self.reset()
        self.difficult_examples = difficult_examples

    def reset(self):
        """Resets the meter with empty member variables"""
        self.scores = torch.FloatTensor(torch.FloatStorage())
        self.targets = torch.LongTensor(torch.LongStorage())
        self.filenames = []

    def add(self, output, target, filename):
        """
        Adds new data to the meter.
        Args:
            output (Tensor): NxK tensor with model output scores.
            target (Tensor): Binary NxK tensor with ground truth labels.
            filename (list): List of filenames corresponding to the data.
        """
        if not torch.is_tensor(output):
            output = torch.from_numpy(output)
        if not torch.is_tensor(target):
            target = torch.from_numpy(target)

        # Ensure correct dimensions for output and target
        if output.dim() == 1:
            output = output.view(-1, 1)
        else:
            assert output.dim() == 2, \
                'wrong output size (should be 1D or 2D with one column per class)'
        if target.dim() == 1:
            target = target.view(-1, 1)
        else:
            assert target.dim() == 2, \
                'wrong target size (should be 1D or 2D with one column per class)'
        if self.scores.numel() > 0:
            assert target.size(1) == self.targets.size(1), \
                'dimensions for output should match previously added examples.'

        # Resize storage if needed
        if self.scores.storage().size() < self.scores.numel() + output.numel():
            new_size = math.ceil(self.scores.storage().size() * 1.5)
            self.scores.storage().resize_(int(new_size + output.numel()))
            self.targets.storage().resize_(int(new_size + output.numel()))

        # Store scores and targets
        offset = self.scores.size(0) if self.scores.dim() > 0 else 0
        self.scores.resize_(offset + output.size(0), output.size(1))
        self.targets.resize_(offset + target.size(0), target.size(1))
        self.scores.narrow(0, offset, output.size(0)).copy_(output)
        self.targets.narrow(0, offset, target.size(0)).copy_(target)

        # Record filenames
        self.filenames += filename

    def value(self):
        """
        Returns the model's average precision for each class.
        Return:
            ap (FloatTensor): 1xK tensor with average precision for each class.
        """
        if self.scores.numel() == 0:
            return 0
        ap = torch.zeros(self.scores.size(1))
        rg = torch.arange(1, self.scores.size(0)).float()
        # Compute average precision for each class
        for k in range(self.scores.size(1)):
            scores = self.scores[:, k]
            targets = self.targets[:, k]
            ap[k] = AveragePrecisionMeter.average_precision(scores, targets, self.difficult_examples)
        return ap

    @staticmethod
    def average_precision(output, target, difficult_examples=True):
        """
        Computes the average precision for a single class.
        Args:
            output (Tensor): Model scores for the class.
            target (Tensor): Ground truth labels for the class.
        """
        device = output.device
        target = target.to(device)
        
        # Sort examples by score in descending order
        sorted, indices = torch.sort(output, dim=0, descending=True)

        # Compute precision at each rank
        pos_count = 0.
        total_count = 0.
        precision_at_i = 0.
        for i in indices:
            label = target[i]
            if difficult_examples and label == 0:
                continue
            if label == 1:
                pos_count += 1
            total_count += 1
            if label == 1:
                precision_at_i += pos_count / total_count
        if pos_count > 0:
            precision_at_i /= pos_count
        else:
            precision_at_i = 0.0
        return precision_at_i

    def overall(self):
        """
        Computes overall metrics (OP, OR, OF1, CP, CR, CF1).
        """
        if self.scores.numel() == 0:
            return 0
        scores = self.scores.cpu().numpy()
        targets = self.targets.clone().cpu().numpy()
        targets[targets == -1] = 0
        return self.evaluation(scores, targets)

    def overall_topk(self, k):
        """
        Computes overall metrics using top-k predictions.
        Args:
            k (int): Number of top predictions to consider.
        """
        targets = self.targets.clone().cpu().numpy()
        targets[targets == -1] = 0
        n, c = self.scores.size()
        scores = np.zeros((n, c)) - 1
        index = self.scores.topk(k, 1, True, True)[1].cpu().numpy()
        tmp = self.scores.cpu().numpy()
        for i in range(n):
            for ind in index[i]:
                scores[i, ind] = 1 if tmp[i, ind] >= 0 else -1
        return self.evaluation(scores, targets)

    def evaluation(self, scores_, targets_):
        """
        Evaluates the model's performance using precision, recall, and F1-score.
        Args:
            scores_ (ndarray): Predicted scores.
            targets_ (ndarray): Ground truth labels.
        """
        # Initialize variables to store counts
        n, n_class = scores_.shape
        Nc, Np, Ng = np.zeros(n_class), np.zeros(n_class), np.zeros(n_class)
        for k in range(n_class):
            scores = scores_[:, k]
            targets = targets_[:, k]
            targets[targets == -1] = 0
            Ng[k] = np.sum(targets == 1)  # Ground truth positives
            Np[k] = np.sum(scores >= 0)  # Predicted positives
            Nc[k] = np.sum(targets * (scores >= 0))  # Correctly predicted positives
        Np[Np == 0] = 1  # Avoid division by zero
        OP = np.sum(Nc) / np.sum(Np)  # Overall precision
        OR = np.sum(Nc) / np.sum(Ng)  # Overall recall
        OF1 = (2 * OP * OR) / (OP + OR)  # Overall F1-score

        CP = np.sum(Nc / Np) / n_class  # Class-wise precision
        CR = np.sum(Nc / Ng) / n_class  # Class-wise recall
        CF1 = (2 * CP * CR) / (CP + CR)  # Class-wise F1-score
        return OP, OR, OF1, CP, CR, CF1