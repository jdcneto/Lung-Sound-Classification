import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

def CB_loss(device, samples_per_cls, no_of_classes, beta):
    """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      samples_per_cls: A python list of size [no_of_classes].
      no_of_classes: total number of classes. int
      loss_type: string. One of "sigmoid", "focal", "softmax".
      beta: float. Hyperparameter for Class balanced loss.
      gamma: float. Hyperparameter for Focal loss.
    Returns:
      cb_loss: A float tensor representing class balanced loss
    """
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = torch.from_numpy(weights / np.sum(weights)*no_of_classes)
    weights = weights.float().to(device)
    loss_weighted = nn.CrossEntropyLoss(weight=weights)
    #cb_loss = loss_weighted(logits, labels)
    
    # retuning criterion
    return loss_weighted