# regularizers.py

def l1_loss(model, loss, lambda_l1):
  l1 = 0
  for p in model.parameters():
    l1 = l1 + p.abs().sum()
  loss = loss + lambda_l1 * l1
  return loss
