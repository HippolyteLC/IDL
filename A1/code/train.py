import torch


VAL_OR_TEST_STRING = "Validation"


def init_loss_and_error(device):
    total_loss = torch.zeros((), device=device)
    total_correct = torch.zeros((), device=device)
    return total_loss, total_correct


def clock_train(data_loader, optimizer, criterion, device, scaler):
    """
    cm: is the clock method 
    """
    total_loss, total_correct = init_loss_and_error(device)  
    for _, (X, y) in enumerate(data_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        X, y = X.to(device), y.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        with torch.cuda.amp.autocast():
            loss, error = criterion.forward(X, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.detach() * X.size(0)
        total_correct += error.sum()
        
    total_loss = total_loss.item()/len(data_loader.dataset)
    total_correct = total_correct.item()/len(data_loader.dataset)

    print(f"Train Error: \n Avg mins off: {(total_correct):>0.1f}, Avg loss: {total_loss:>8f} \n")

    return total_loss, total_correct    


def clock_val(data_loader, criterion, device):
    total_loss, total_correct = init_loss_and_error(device)  

    with torch.no_grad():
        for _, (X, y) in enumerate(data_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            X, y = X.to(device), y.to(device)

            # forward + backward + optimize
            loss, error = criterion.forward(X, y)

            total_loss += loss.detach() * X.size(0)
            total_correct += error.sum()
        
    total_loss = total_loss.item()/len(data_loader.dataset)
    total_correct = total_correct.item()/len(data_loader.dataset)

    print(f"{VAL_OR_TEST_STRING} Error: \n Avg mins off: {(total_correct):>0.1f}, Avg loss: {total_loss:>8f} \n")

    return total_loss, total_correct    


def main_train(data_loader, optimizer, criterion, device, scaler):
    total_loss, total_correct = init_loss_and_error(device)  

    for _, (X, y) in enumerate(data_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        X, y = X.to(device), y.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        with torch.amp.autocast('cuda'):
            loss, error = criterion.forward(X, y)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.detach() * X.size(0)
        total_correct += error.sum()
    
    total_loss = total_loss.item()/len(data_loader.dataset)
    total_correct = total_correct.item()/len(data_loader.dataset)

    print(f"Train Error: \n Accuracy: {(100*total_correct):>0.1f}%, Avg loss: {total_loss:>8f} \n")

    return total_loss, total_correct


def main_val(data_loader, criterion, device):
    total_loss, total_correct = init_loss_and_error(device)  

    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            loss, error = criterion.forward(X, y)
            total_loss += loss * X.size(0)
            total_correct += error.sum() 

    total_loss = total_loss / len(data_loader.dataset)
    total_correct = total_correct / len(data_loader.dataset)

    print(f"{VAL_OR_TEST_STRING} Error: \n Accuracy: {(100*total_correct):>0.1f}%, Avg loss: {total_loss:>8f} \n")
    
    return total_loss, total_correct
