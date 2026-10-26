import torch
import torch.nn as nn

from converters import get_label_converter_fn, output_to_label

class CyclicRegressionLoss(nn.Module):
    def forward(self, pred, labels):
        """ 
        Assumes scaled output between 0 and 1.
        """
        return 1 - torch.cos((pred-labels)*2*torch.pi)

def get_loss_fc(method: int):
    return {
        1: tuple(nn.CrossEntropyLoss()), 
        2: tuple(CyclicRegressionLoss()),
        3: (nn.CrossEntropyLoss(), nn.CrossEntropyLoss()),
        4: (nn.CrossEntropyLoss(), CyclicRegressionLoss()),
        5: (CyclicRegressionLoss(), nn.CrossEntropyLoss()),
        6: (CyclicRegressionLoss(), CyclicRegressionLoss()),
        7: (nn.CosineSimilarity(), nn.CosineSimilarity())
    }.get(method, tuple(nn.CrossEntropyLoss()))

def clock_accuracy(pred, label):
    """
    Common sense accuracy.
    input: pred, label. outputs the shortest distance from pred to label considering
    the full cycle of 720 minutes. 
    I.e. error is in minutes that the prediction is off by. 
    """
    pred_in_mins = pred[:,0]*60 + pred[:,1]
    label_in_mins = label[:,0]*60 + label[:,1]
    diff = torch.abs(pred_in_mins - label_in_mins)
    return torch.minimum(diff, 720 - diff) 

class Criterion:
    """
    This class takes model/ training parameters and flexibly outputs the 
    final loss calculation, the correctly formatted outputs. 
    
    """
    def __init__(self, net, num_output_heads, method):
        self.net = net

        self.loss_funcs = get_loss_fc(method)
        self.output_converter = output_to_label(self.method)

        self.method = method
        self.num_output_heads = num_output_heads

        self.label_converter = None    

    def forward(self, X, y):
        if self.label_converter is None:
            self.label_converter = get_label_converter_fn(self.method, y.shape[-1])

        outputs = self.net(X) # tuple of outputs if multiple heads
        labels = self.label_converter(y)

        losses = []
        for i in range(self.num_output_heads):
            loss = self.loss_funcs[i](outputs[i], labels[i])
            losses.append(loss)
        loss = torch.sum(losses)

        standardized_output = self.output_converter(outputs)
        if self.method:
            error = clock_accuracy(standardized_output, y)
        else:
            error = (outputs.argmax(1) == y).sum()
        return loss, error
    
