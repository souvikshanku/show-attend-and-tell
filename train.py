import torch
import torch.optim as optim


def train(encoder, decoder,  train_dataloader, num_epochs=1):
    optimizer = optim.Adam(
        [*encoder.parameters(), *decoder.parameters()], lr=0.003
    )
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    hist_loss = []

    for epoch in range(num_epochs):
        print(f" ---------------- Epoch: {epoch + 1} ----------------")

        for imgs, labels in train_dataloader:
            imgs = imgs.clone().detach().float()

            img_features = encoder(imgs)
            out, _ = decoder.forward(img_features, labels[:, :-1, :])

            loss = - torch.sum(out * labels[:, 1:, :]) / out.shape[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            hist_loss.append(loss.item())
            print(loss.item())

        scheduler.step()

        torch.save({
            "epoch": epoch + 1,
            "encoder_state_dict": encoder.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, f"checkpoints/epoch_{epoch + 1}.pt")
        print("Checkpint saved...")

    return hist_loss
