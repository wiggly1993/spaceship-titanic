import torch.nn as nn
import torch
import tqdm
import numpy as np

def training_loop(model: torch.nn.Module, 
                  train_loader: torch.utils.data.Dataset, 
                  test_loader: torch.utils.data.Dataset, 
                  num_epochs: int,
                  show_progress: bool = False):
    

    criterion = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    # send to device 
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)


    # prepare for training loop
    epoch_train_loss = []
    epoch_val_loss = []
    epoch_accuracy = []
    epoch_pbar = tqdm.tqdm(range(num_epochs))
    counter = 0


    for epoch in epoch_pbar:
        batch_train_loss = []
        model.train()
        for minibatch in train_loader:
            X, y = minibatch
            X, y = X.to(device), y.to(device)

            pred = model(X)
            batch_loss = criterion(pred, y)

            opt.zero_grad()
            batch_loss.backward()
            opt.step()

            batch_train_loss.append(round(batch_loss.item(),3))


        batch_val_loss = []
        accuracy_batch = []
        model.eval()
        with torch.no_grad():
            for minibatch in test_loader:
                X, y = minibatch
                X,y = X.to(device), y.to(device)

                pred = model(X)
                batch_loss = criterion(pred, y)

                batch_val_loss.append(round(batch_loss.item(),3))

                preds = (torch.sigmoid(pred) > 0.5).float()
                accuracy = (preds == y).float().mean().item()

                accuracy_batch.append(accuracy)

        
        epoch_train_loss.append(np.average(batch_train_loss))
        epoch_val_loss.append(np.average(batch_val_loss))
        epoch_accuracy.append(np.average(accuracy_batch))


    
        # this still happens once per epoch, we use it to display how the loss is evolving
        epoch_pbar.set_postfix(train_loss=epoch_train_loss[epoch], eval_loss=epoch_val_loss[epoch])
        tqdm.tqdm.write(f"Epoch {epoch} --- train_loss: {epoch_train_loss[epoch]:.2f} \
                        --- eval_loss: {epoch_val_loss[epoch]:.2f} ---accuracy: {epoch_accuracy[epoch]:.2f}")