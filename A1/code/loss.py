import torch
import torch.nn as nn

from converters import get_label_converter_fn, output_to_label

class CosineSimilarityLoss(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, pred, labels):
        """
        Computes 1 - cosine similarity as a loss, averaged over the batch.
        Ensures the final output is a strict scalar.
        """
        similarity = torch.cosine_similarity(pred, labels, dim=self.dim)
        return 1.0 - similarity.mean()

class CyclicRegressionLoss(nn.Module):
    def forward(self, pred, labels):
        """ 
        Assumes scaled output between 0 and 1.
        """
        return (1 - torch.cos((pred-labels)*2*torch.pi)).mean()

def get_loss_fc(method: int):
    return {
        1: (nn.CrossEntropyLoss(),), 
        2: (CyclicRegressionLoss(),),
        3: (nn.CrossEntropyLoss(), nn.CrossEntropyLoss()),
        4: (nn.CrossEntropyLoss(), CyclicRegressionLoss()),
        5: (CyclicRegressionLoss(), nn.CrossEntropyLoss()),
        6: (CyclicRegressionLoss(), CyclicRegressionLoss()),
        7: (CosineSimilarityLoss(), CosineSimilarityLoss())
    }.get(method, (nn.CrossEntropyLoss(),))

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
    def __init__(self, net, model_config):
        self.net = net

        self.method = model_config.clock_method
        self.loss_funcs = get_loss_fc(self.method)
        self.output_converter = output_to_label(self.method)

        self.num_output_heads = model_config.num_output_heads

        self.label_converter = None    

    def forward(self, X, y, verbose=False):
        # TODO: fix for method 3 (cls, cls)
        # TODO: fix for method 2 (train loss and validation loss are constant/ no not change)
        # TODO: fix for method 3, wrong shape (batch/ output dims)
        if self.label_converter is None:
            self.label_converter = get_label_converter_fn(self.method, y.shape[-1])

        outputs = self.net(X) # tuple of outputs if multiple heads
        labels = self.label_converter(y)
        loss = torch.tensor(0.0, device=X.device)
        if verbose:
            print("outputs.shape:", outputs.shape)
            print("labels.shape:", labels.shape)
            print("labels.dtype:", labels.dtype)

            print("labels min:", labels.min().item())
            print("labels max:", labels.max().item())

            num_classes = outputs.shape[1]
            print("num_classes:", num_classes)

            invalid = (labels < 0) | (labels >= num_classes)
            print("Invalid labels:", labels[invalid])

        if self.num_output_heads == 1:
            loss = self.loss_funcs[0](outputs, labels)
        else:
            for i in range(self.num_output_heads):
                head_loss = self.loss_funcs[i](outputs[i], labels[i])
                if head_loss.ndim > 0:
                    head_loss = head_loss.mean()
                loss = loss + head_loss

        standardized_output = self.output_converter(outputs)
        if self.method:
            error = clock_accuracy(standardized_output, y)
        else:
            error = (outputs.argmax(1) == y).sum()
        return loss, error
    
