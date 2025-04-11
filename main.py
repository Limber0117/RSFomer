import torch
from sklearn.model_selection import StratifiedKFold
from dataset import Dataset
from RSFormer import RSFormer
from process import Trainer
import torch.utils.data as Data
from args import Train_data, Test_data
from datautils import *

# User custom dataset
def main():
    X = np.concatenate((Train_data[0], Test_data[0]), axis=0)
    y = np.concatenate((Train_data[1], Test_data[1]), axis=0)
    X = filter(X)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    torch.set_num_threads(6)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        mean, std = mean_standardize_fit(X_train)
        X_train, X_test = mean_standardize_transform(X_train, mean, std), mean_standardize_transform(X_test, mean, std)
        train_dataset = Dataset(X_train, y_train, device=args.device, mode='train')
        test_dataset = Dataset(X_test, y_test, device=args.device, mode='test')
        train_loader = Data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
        test_loader = Data.DataLoader(test_dataset, batch_size=args.test_batch_size)
        args.data_shape = train_dataset.shape()
        print(args.data_shape)
        print('dataset initial ends')
        model = RSFormer(args)
        print('model initial ends')
        trainer = Trainer(args, model, train_loader, test_loader, verbose=True)
        trainer.train()


if __name__ == '__main__':
    main()
