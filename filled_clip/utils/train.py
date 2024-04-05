from utils import config

from tqdm.auto import tqdm


def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
    total_loss = 0

    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        batch = {k: v.to(config.device) for k, v in batch.items() if k != "caption"}
        # print(batch["image"].shape)
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()

        total_loss += loss.item()

        tqdm_object.set_postfix(train_loss=loss.item())

    return total_loss / len(train_loader)


def valid_epoch(model, valid_loader):
    total_loss = 0

    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:
        batch = {k: v.to(config.device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)

        total_loss += loss.item()

        tqdm_object.set_postfix(valid_loss=loss.item())
    return total_loss / len(valid_loader)
