import torch
from dataset import Dataset
from RSFormer import RSFormer
from process import Trainer
import torch.utils.data as Data
from args import Train_data, Test_data
from datautils import *

# UEA dataset: set data_path = None
def UEA_main():
    torch.set_num_threads(6)
    X_train, X_test = Train_data[0], Test_data[0]
    y_train, y_test = Train_data[1], Test_data[1]
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
    UEA_main()