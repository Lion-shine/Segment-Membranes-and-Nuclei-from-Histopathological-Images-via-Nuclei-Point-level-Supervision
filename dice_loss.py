import torch
from torch.autograd import Function
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
class BEC_Jaccard_Loss_softmax:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.nll_loss = nn.CrossEntropyLoss()
    def __call__(self, outputs, targets):
        eps = 1e-15
        # jaccard_target = (targets>=0.1).float()
        jaccard_target = targets
        jaccard_output = torch.softmax(outputs, dim=1)
        # import pdb; pdb.set_trace()
        intersection = (jaccard_output * jaccard_target).sum()
        union = jaccard_output.sum() + jaccard_target.sum()
        # import pdb; pdb.set_trace()
        loss = self.alpha * self.nll_loss(outputs, targets.max(1)[1])
        loss -= (1-self.alpha) * torch.log((intersection+eps)/(union-intersection+eps))
        return loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input) # 这里转成log(pt)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


class BEC_Jaccard_Loss:
    def __init__(self, alpha=0.8):
        self.alpha = alpha
        self.nll_loss = nn.BCEWithLogitsLoss()
    def __call__(self, outputs, targets):
        eps = 1e-15
        # jaccard_target = (targets>=0.1).float()
        jaccard_target = (targets).float()
        jaccard_output = torch.sigmoid(outputs)
        # import pdb; pdb.set_trace()
        intersection = (jaccard_output * jaccard_target).sum()
        union = jaccard_output.sum() + jaccard_target.sum()
        loss = self.alpha * self.nll_loss(outputs, targets)
        loss -= (1-self.alpha) * torch.log((intersection+eps)/(union-intersection+eps))
        return loss

class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)