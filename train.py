# train.py

from tqdm import tqdm
import torch.nn.functional as F

from . import models

def train(model, device, train_loader, optimizer, epoch, opts, hyper_params):
  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0

  train_losses = []
  train_acc = []

  for batch_idx, (data, target) in enumerate(pbar):
    # get samples
    data, target = data.to(device), target.to(device)

    # Init
    optimizer.zero_grad()
    # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
    # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

    # Predict
    y_pred = model(data)

    # Calculate loss
    loss = F.nll_loss(y_pred, target)
    # L1 Regularization
    if models.ModelOpts.L1 in opts:
      l1_lambda = hyper_params.get("l1_lambda", 0.1)
      loss = l1_loss(model, loss, l1_lambda)

    train_losses.append(loss)

    # Backpropagation
    loss.backward()
    optimizer.step()

    # Update pbar-tqdm
    
    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)

    pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    train_acc.append(100*correct/processed)

  return train_losses, train_acc
