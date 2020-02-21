import copy
import torch
import torch.nn.functional as F


def train_model(model, optimizer, train_loader, test_loader, device, max_patience=5):
    best_model = copy.deepcopy(model.cuda())
    best_model = best_model.cuda()
    best_loss = 1e6

    for _ in range(100):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                with torch.no_grad():
                    test_loss = 0
                    for test_idx, (test_data, test_target) in enumerate(test_loader):
                        test_data, test_target = test_data.to(device), test_target.to(device)
                        test_output = model(test_data)
                        test_loss += F.nll_loss(test_output, test_target)

                print(test_loss)
                if test_loss < best_loss:
                    best_model = copy.deepcopy(model)
                    best_loss = test_loss
                    patience = 0
                else:
                    patience += 1

            if patience >= max_patience:
                break
        if patience >= max_patience:
            break

    return best_model.cuda()