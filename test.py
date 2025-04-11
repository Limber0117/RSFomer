import torch
from dataset import Dataset
from RSFormer import RSFormer
import torch.utils.data as Data
import json
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import argparse
import os


class Args:
    def __init__(self, args_dict):
        self.__dict__.update(args_dict)


def load_args_from_json(json_path):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Config file not found: {json_path}")
    with open(json_path, 'r') as f:
        args = json.load(f)
    return args


def load_model(model_path, args):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = RSFormer(args)
    model.load_state_dict(torch.load(model_path, map_location=args.device))
    model.to(args.device)
    model.eval()
    return model


def evaluate(model, dataloader, device):
    all_preds = []
    all_labels = []
    tqdm_data_loader = tqdm(dataloader)

    with torch.no_grad():
        for idx, batch in enumerate(tqdm_data_loader):
            seqs, label = [x.to(device) for x in batch]
            seqs = seqs.float()
            scores = model(seqs, mode='test')
            pred = scores.argmax(dim=1)
            all_preds.append(pred.detach().cpu().numpy())
            all_labels.append(label.detach().cpu().numpy())

    y_true = np.concatenate(all_labels, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)

    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    f1 = f1_score(y_true=y_true, y_pred=y_pred, average='macro')

    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test F1:       {f1:.4f}")


def inference(json_path, model_path):
    args_dict = load_args_from_json(json_path)
    args = Args(args_dict)

    X_test = np.load(os.path.join(args.data_path, 'X_test.npy'),
                     allow_pickle=True).astype(np.float32)
    y_test = np.load(os.path.join(args.data_path, 'y_test.npy'),
                     allow_pickle=True).astype(np.float32)

    test_dataset = Dataset(X_test, y_test, device=args.device, mode='test')
    test_loader = Data.DataLoader(test_dataset, batch_size=args.test_batch_size)

    args.data_shape = test_dataset.shape()
    print("Data shape:", args.data_shape)

    print(f"Loading model from {model_path}...")
    model = load_model(model_path, args)

    print("Running inference...")
    evaluate(model, test_loader, device=args.device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str, default='./test/args.json')
    parser.add_argument('--model_path', type=str, default='./test/model.pkl')
    cmd_args = parser.parse_args()
    inference(cmd_args.json_path, cmd_args.model_path)
