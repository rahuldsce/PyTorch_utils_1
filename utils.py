import sys
import matplotlib.pyplot as plt

def plot_results(train_losses, train_acc, test_losses, test_acc):
  fig, axs = plt.subplots(2,2,figsize=(15,10))
  for label, losses in train_losses.items():
    axs[0, 0].plot(losses, label = label)
  axs[0, 0].set_title("Training Loss")
  axs[0, 0].legend()

  for label, acc in train_acc.items():
    axs[1, 0].plot(acc, label = label)
  axs[1, 0].set_title("Training Accuracy")
  axs[1, 0].legend()

  for label, losses in test_losses.items():
    axs[0, 1].plot(losses, label = label)
  axs[0, 1].set_title("Test Loss")
  axs[0, 1].legend()

  for label, acc in test_acc.items():
    axs[1, 1].plot(acc, label = label)
  axs[1, 1].set_title("Test Accuracy")
  axs[1, 1].legend()

def printout(msg):
  print(msg)
  sys.stdout.flush()

def show_images_from_loader(data_loader, image_count=60):
  figure = plt.figure()
  for data, target in data_loader:
    for index in range(data.shape[0]):
      plt.subplot(6, 10, index+1)
      plt.axis('off')
      plt.imshow(data[index].numpy().squeeze(), cmap='gray_r')
      if index + 1 == image_count:
        break
    break

# Show misclassified images with respect to a model
def show_misclassified_images_from_model(model, model_path, data_loader, class_labels, image_count):
  model.load_state_dict(torch.load(model_path))
  model.to(device)
  model.eval()

  correct = 0
  figure = plt.figure(figsize=(15,15))
  count = 0
  with torch.no_grad():
      for data, target in data_loader:
          data, target = data.to(device), target.to(device)
          output = model(data)
          pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
          correct += pred.eq(target.view_as(pred)).sum().item()

          for idx in range(len(pred)):
            i_pred, i_act = pred[idx], target[idx]
            if i_pred != i_act:
                annotation = "Actual: %s, Predicted: %s" % (class_labels[i_act], class_labels[i_pred])
                count += 1
                plt.subplot(image_count/5, 5, count)
                plt.axis('off')
                plt.imshow(data[idx].cpu().numpy().squeeze(), cmap='gray_r')
                plt.annotate(annotation, xy=(0,0), xytext=(0,-1.2), fontsize=13)
            if count == image_count:
                return
