import torch
from torch import nn
from eval import _validate
from models.cnn import CustomNet


def _train(epoch, model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()

        preds = model(inputs)
        loss = criterion(preds, targets)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = preds.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total
    print(f'Train Epoch: {epoch} Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}%')


def train_model(train_loader, val_loader):

    model = CustomNet().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_acc = 0
    # Run the training process for {num_epochs} epochs
    num_epochs = 10
    for epoch in range(1, num_epochs + 1):
        _train(epoch, model, train_loader, criterion, optimizer)

        # At the end of each training iteration, perform a validation step
        val_accuracy = _validate(model, val_loader, criterion)

        # Best validation accuracy
        best_acc = max(best_acc, val_accuracy)


    print(f'Best validation accuracy: {best_acc:.2f}%')