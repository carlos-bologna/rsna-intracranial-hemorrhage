import torch
import numpy as np

class QuadraticKappa(object):

    def __init__(self, n_classes, eps=1e-10):
        self.n_classes = n_classes
        self.eps = eps
        # Detect if we have a GPU available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __call__(self, p, y):
        """
        QWK loss function as described in https://arxiv.org/pdf/1612.00775.pdf

        Arguments:
            p: a tensor with probability predictions, [batch_size, n_classes],
            y, a tensor with one-hot encoded class labels, [batch_size, n_classes]
        Returns:
            QWK loss
        """
        _, p = torch.max(p, 1)
        _, y = torch.max(y, 1)

        #p = p.cpu().type(torch.float32)
        #y = y.cpu().type(torch.float32)
        #p = torch.from_numpy(p.cpu().numpy().astype(np.float32), requires_grad=True)
        #y = torch.from_numpy(y.cpu().numpy().astype(np.float32), requires_grad=True)

        p = torch.tensor(p.cpu().numpy().astype(np.float32), requires_grad=True)
        y = torch.tensor(y.cpu().numpy().astype(np.float32), requires_grad=True)

        W = np.zeros((self.n_classes, self.n_classes))
        for i in range(self.n_classes):
            for j in range(self.n_classes):
                W[i,j] = (i-j)**2

        #W = torch.from_numpy(W.astype(np.float32)).to(self.device)
        W = torch.from_numpy(W.astype(np.float32))

        O = torch.matmul(y.t(), p)
        E = torch.matmul(y.sum(dim=0).view(-1,1), p.sum(dim=0).view(1,-1)) / O.sum()
        K = (W*O).sum() / ((W*E[0][0]).sum() + self.eps)
        return K.to(self.device)

class WeightedMultiLabelLogLoss(object):

    def __init__(self, n_classes, weight):
        self.n_classes = n_classes
        self.weight = weight
        # Detect if we have a GPU available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def __call__(self, p, y):
        """
        Weighted Log Loss function for multiple class prediction.

        Arguments:
            p: a tensor with probability predictions, [batch_size, n_classes],
            y, a tensor with one-hot encoded class labels, [batch_size, n_classes]
        Returns:
            Log loss
        """
        #p = p.cpu()
        #y = y.cpu()

        trues = y.type(torch.float)
        #w = torch.from_numpy(self.weight).type(torch.float)
        w = self.weight.type(torch.float)

        preds = torch.clamp(p, 1e-7, (1-1e-7))

        loss_subtypes = trues.type(torch.float) * torch.log(preds) + (1.0 - trues) * torch.log(1.0 - preds)
        loss_weighted = torch.mean((loss_subtypes * w), dim=1)
        loss = - torch.mean(loss_weighted)

        #return loss.to(self.device)
        return loss

class WeightedMultiLabelFocalLogLoss(object):

    def __init__(self, n_classes, weight, gamma=2):
        self.n_classes = n_classes
        self.weight = weight
        self.gamma = gamma
        # Detect if we have a GPU available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def __call__(self, p, y):
        """
        Weighted Focal Log Loss function for multiple class prediction.

        Arguments:
            p: a tensor with probability predictions, [batch_size, n_classes],
            y, a tensor with one-hot encoded class labels, [batch_size, n_classes]
        Returns:
            Log loss
        """
        #p = p.cpu()
        #y = y.cpu()

        trues = y.type(torch.float)
        #w = torch.from_numpy(self.weight).type(torch.float)
        w = self.weight.type(torch.float)

        preds = torch.clamp(p, 1e-7, (1-1e-7))

        loss_subtypes = trues.type(torch.float) * torch.log(preds) + (1.0 - trues) * torch.log(1.0 - preds)
        fix_weights = (1 - preds) ** self.gamma
        focal_loss = (loss_subtypes * w) * fix_weights
        loss_weighted = torch.mean(focal_loss, dim=1)
        loss = - torch.mean(loss_weighted)

        #return loss.to(self.device)
        return loss
