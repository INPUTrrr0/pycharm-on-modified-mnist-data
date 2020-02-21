import torch
import numpy as np


def get_prediction(net, valid_loader, device):
    preds = []
    for batch_idx, (data, _) in enumerate(valid_loader):
        with torch.no_grad():
            test_data = data.to(device)
            test_output = net(test_data)
            test_output = test_output.cpu()
            #test_output = test_output.argmax()
            test_output = test_output.numpy()
            pred = np.argmax(test_output, 1)
            preds.append(pred)

    return np.concatenate(preds)
