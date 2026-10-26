import torch

from utils import clock_accuracy, get_regularization_term

def clock_loss(X, y, net, num_output_heads, label_converter, loss_fn):
    if isinstance(loss_fn, tuple) and num_output_heads == 2:
        loss_fn_1, loss_fn_2 = loss_fn
    if num_output_heads == 1:
        outputs = net(X)
        labels = label_converter(y)
        assert outputs.shape[0] == labels.shape[0], f"output shape {outputs.shape} is not same as label shape {labels.shape}"
        loss = loss_fn(outputs, labels)
    elif num_output_heads == 2:
        outputs = net(X)
        output_1, output_2 = outputs[:,0], outputs[:,1]
        lab_1, lab_2 = label_converter(y)    
        loss_1, loss_2 = loss_fn_1(output_1, lab_1), loss_fn_2(output_2, lab_2)
        loss = loss_1 + loss_2
    return outputs, loss


def clock_train(data_loader, optimizer, net, loss_fn, num_output_heads, \
                output_converter, label_converter, device):
    """
    cm: is the clock method 
    """
    size = len(data_loader.dataset)
    total_loss = torch.zeros((), device=device)
    total_correct = torch.zeros((), device=device)
    for _, (X, y) in enumerate(data_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        X, y = X.to(device), y.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs, loss = clock_loss(X, y, net, num_output_heads, label_converter, loss_fn)

        loss.backward()
        optimizer.step()

        total_loss += loss.detach() * X.size(0)
        outputs_correct_format = output_converter(outputs)
        error = clock_accuracy(outputs_correct_format, y)
        total_correct += error.sum()
        
    total_loss = total_loss.item()/size
    total_correct = total_correct.item()/size

    print(f"Train Error: \n Avg mins off: {(total_correct):>0.1f}, Avg loss: {total_loss:>8f} \n")

    return total_loss, total_correct    

def clock_val(data_loader, net, loss_fn, num_output_heads, output_converter, label_converter, device):
    size = len(data_loader.dataset)
    total_loss = torch.zeros((), device=device)
    total_correct = torch.zeros((), device=device)
    with torch.no_grad():
        for _, (X, y) in enumerate(data_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            X, y = X.to(device), y.to(device)

            # forward + backward + optimize
            outputs, loss = clock_loss(X, y, net, num_output_heads, label_converter, loss_fn)

            total_loss += loss.detach() * X.size(0)

            outputs_correct_format = output_converter(outputs)
            error = clock_accuracy(outputs_correct_format, y)
            total_correct += error.sum()
        
    total_loss = total_loss.item()/size
    total_correct = total_correct.item()/size

    print(f"Val Error: \n Avg mins off: {(total_correct):>0.1f}, Avg loss: {total_loss:>8f} \n")

    return total_loss, total_correct    

def clock_test(data_loader, net, loss_fn, num_output_heads, output_converter, label_converter, device):
    """
    cm: is the clock method 
    """
    size = len(data_loader.dataset)
    total_loss = torch.zeros((), device=device)
    total_correct = torch.zeros((), device=device)
    with torch.no_grad():
        for _, (X, y) in enumerate(data_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            X, y = X.to(device), y.to(device)

            # forward + backward + optimize
            outputs, loss = clock_loss(X, y, net, num_output_heads, label_converter, loss_fn)

            total_loss += loss.detach() * X.size(0)

            outputs_correct_format = output_converter(outputs)
            error = clock_accuracy(outputs_correct_format, y)
            total_correct += error.sum()
        
    total_loss = total_loss.item()/size
    total_correct = total_correct.item()/size

    print(f"Test Error: \n Avg mins off: {(total_correct):>0.1f}, Avg loss: {total_loss:>8f} \n")

    return total_loss, total_correct    

def main_train(data_loader, optimizer, net, loss_fn, regularization_term, device):
    size = len(data_loader.dataset)
    total_loss = torch.zeros((), device=device)
    total_correct = torch.zeros((), device=device)
    for _, (X, y) in enumerate(data_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        X, y = X.to(device), y.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(X)
        loss = loss_fn(outputs, y)
        reg_term = get_regularization_term(regularization_term, net) # compute l2 (or other) regularization terms
        lambda_reg=0.01 # for loss regularization terms
        loss_reg = loss + lambda_reg*reg_term

        loss_reg.backward()
        optimizer.step()

        total_loss += loss.detach() * X.size(0)
        total_correct += (outputs.argmax(1) == y).sum()
    
    total_loss = total_loss.item()/size
    total_correct = total_correct.item()/size

    print(f"Train Error: \n Accuracy: {(100*total_correct):>0.1f}%, Avg loss: {total_loss:>8f} \n")

    return total_loss, total_correct

def main_val(data_loader, net, loss_fn, device):
    size = len(data_loader.dataset)
    total_loss = torch.zeros((), device=device)
    total_correct = torch.zeros((), device=device)
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            outputs = net(X)
            total_loss += loss_fn(outputs, y) * X.size(0)
            total_correct += (outputs.argmax(1) == y).sum() 
    total_loss = total_loss / size
    total_correct = total_correct / size
    print(f"Validation Error: \n Accuracy: {(100*total_correct):>0.1f}%, Avg loss: {total_loss:>8f} \n")
    return total_loss, total_correct

def main_test(data_loader, net, loss_fn, device):
    size = len(data_loader.dataset)
    total_loss = torch.zeros((), device=device)
    total_correct = torch.zeros((), device=device)
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            outputs = net(X)
            total_loss += loss_fn(outputs, y) * X.size(0)
            total_correct += (outputs.argmax(1) == y).sum() 
    total_loss/=size
    total_correct/=size
    print(f"Test Error: \n Accuracy: {(100*total_correct):>0.1f}%, Avg loss: {total_loss:>8f} \n")
    return total_loss, total_correct