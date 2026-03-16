import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from torch.utils.data import DataLoader, random_split

from dataset.loader import Hot3DDataset
from models.baseline_model import SingleViewBaseline
from models.multiview_model import MultiViewConcat
from tqdm import tqdm


def main():

    # "single" or "multi_avg" or "multi_concat"
    MODE = "multi_avg"

    # load full dataset
    dataset = Hot3DDataset("./dataset/train_quest3", mode=MODE)

    # split into train / validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # dataloaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    #from our baseline_model.py
    if MODE == "single":
        model = SingleViewBaseline(output_dim=22)

    elif MODE == "multi_avg":
        model = SingleViewBaseline(output_dim=22)

    elif MODE == "multi_concat":
        model = MultiViewConcat(output_dim=22)
    
    #loss function (mean squared error)
    criterion = torch.nn.MSELoss()
    #training optimizer using Adam. learning rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    #epochs = loops over the dataset
    num_epochs = 10
    #record best loss seen
    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        # training
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"{MODE} Epoch {epoch+1}/{num_epochs} [Train]")

        for batch in train_bar:
            targets = batch["target"]

            if MODE == "single":
                images = batch["image"]
                preds = model(images)

            elif MODE == "multi_avg":
                image_left = batch["image_left"]
                image_right = batch["image_right"]

                #forward pass (prediction)
                pred_left = model(image_left)
                pred_right = model(image_right)

                # naive average fusion
                preds = (pred_left + pred_right) / 2

            elif MODE == "multi_concat":
                image_left = batch["image_left"]
                image_right = batch["image_right"]
                preds = model(image_left, image_right)

            loss = criterion(preds, targets)

            #clear gradient, optimize based on loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #update loss
            running_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())

        avg_train_loss = running_loss / len(train_loader)
        # validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"{MODE} Epoch {epoch+1}/{num_epochs} [Val]")

            for batch in val_bar:
                targets = batch["target"]

                if MODE == "single":
                    images = batch["image"]
                    preds = model(images)

                elif MODE == "multi_avg":
                    image_left = batch["image_left"]
                    image_right = batch["image_right"]

                    #forward pass (prediction)
                    pred_left = model(image_left)
                    pred_right = model(image_right)

                    # naive average fusion
                    preds = (pred_left + pred_right) / 2

                elif MODE == "multi_concat":
                    preds = model(image_left, image_right)
            
                loss = criterion(preds, targets)

                val_loss += loss.item()
                val_bar.set_postfix(loss=loss.item())

        avg_val_loss = val_loss / len(val_loader)
        #check if new best loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"New best val loss: {best_val_loss:.4f}")

        print(
            f"Epoch {epoch+1}: "
            f"Train Loss = {avg_train_loss:.4f}, "
            f"Val Loss = {avg_val_loss:.4f}"
        )


if __name__ == "__main__":
    main()