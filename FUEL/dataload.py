import os
import random
import numpy as np 
from utils import read_data
import torch
from torch.utils.data import Dataset, DataLoader
import pdb

class Federated_Dataset(Dataset):
    def __init__(self, X, Y, A):
        self.X = X
        self.Y = Y
        self.A = A

    def __getitem__(self, index):
        X = self.X[index]
        Y = self.Y[index]
        A = self.A[index]
        return X, Y, A 

    def __len__(self):
        return self.X.shape[0]


#### adult dataset x("51 White", "52 Asian-Pac-Islander", "53 Amer-Indian-Eskimo", "54 Other", "55 Black", "56 Female", "57 Male")
def LoadDataset(args):
    clients_name, groups, train_data, test_data = read_data(args.train_dir, args.test_dir)

    # client_name [phd, non-phd]
    client_train_loads = []
    client_test_loads = []
    args.n_clients = len(clients_name)
    # clients_name = clients_name[:1]
    if args.dataset == "adult":
        for client in clients_name:
            X = np.array(train_data[client]["x"]).astype(np.float32)

            Y = np.array(train_data[client]["y"]).astype(np.float32)

            if args.sensitive_attr == "race":
                A = X[:,51] # [1: white, 0: other]
                X = np.delete(X, [51, 52, 53, 54, 55], axis = 1)
                args.n_feats = X.shape[1]
            elif args.sensitive_attr == "sex":
                A = X[:, 56] # [1: female, 0: male]
                X = np.delete(X, [56, 57], axis = 1)
                args.n_feats = X.shape[1]
            elif args.sensitive_attr == "none-race":
                A = X[:, 51]  # [1: white, 0: other]
                args.n_feats = X.shape[1]
            elif args.sensitive_attr == "none-sex":
                A = X[:, 56]
                args.n_feats = X.shape[1]
            else:
                print("error sensitive attr")
                exit()
            dataset = Federated_Dataset(X, Y, A)
            client_train_loads.append(DataLoader(dataset, X.shape[0],
            shuffle = args.shuffle,
            num_workers = args.num_workers,
            pin_memory = True,
            drop_last = args.drop_last))


        for client in clients_name:
            X = np.array(test_data[client]["x"]).astype(np.float32)
            Y = np.array(test_data[client]["y"]).astype(np.float32)
            if args.sensitive_attr =="race":
                A = X[:,51] # [1: white, 0: other]
                X = np.delete(X, [51, 52, 53, 54, 55],axis = 1)
            elif args.sensitive_attr == "sex":
                A = X[:, 56] # [1: female, 0: male]
                X = np.delete(X, [56, 57], axis = 1)
            elif args.sensitive_attr == "none-race":
                A = X[:, 51]  # [1: white, 0: other]
                args.n_feats = X.shape[1]
            elif args.sensitive_attr == "none-sex":
                A = X[:, 56]
                args.n_feats = X.shape[1]
            else:
                print("error sensitive attr")
                exit()

            dataset = Federated_Dataset(X, Y, A)

            client_test_loads.append(DataLoader(dataset, X.shape[0],
            shuffle = args.shuffle,
            num_workers = args.num_workers,
            pin_memory = True,
            drop_last = args.drop_last)) 

    elif "eicu" in args.dataset:
    # elif args.dataset == "eicu_d" or args.dataset == "eicu_los":
        for client in clients_name:
            X = np.array(train_data[client]["x"]).astype(np.float32)

            Y = np.array(train_data[client]["y"]).astype(np.float32)

            if args.sensitive_attr == "race":
                A = train_data[client]["race"]
                args.n_feats = X.shape[1]
            elif args.sensitive_attr == "sex":
                A = train_data[client]["gender"]
                args.n_feats = X.shape[1]
            else:
                A = train_data[client]["race"]
                args.n_feats = X.shape[1]
            dataset = Federated_Dataset(X, Y, A)
            client_train_loads.append(DataLoader(dataset, X.shape[0],
            shuffle = args.shuffle,
            num_workers = args.num_workers,
            pin_memory = True,
            drop_last = args.drop_last))

        for client in clients_name:
            X = np.array(test_data[client]["x"]).astype(np.float32)
            Y = np.array(test_data[client]["y"]).astype(np.float32)
            if args.sensitive_attr =="race":
                A = test_data[client]["race"]
            elif args.sensitive_attr == "sex":
                A = test_data[client]["gender"]
            else:
                A = test_data[client]["race"]

            dataset = Federated_Dataset(X, Y, A)

            client_test_loads.append(DataLoader(dataset, X.shape[0],
            shuffle = args.shuffle,
            num_workers = args.num_workers,
            pin_memory = True,
            drop_last = args.drop_last)) 

    elif args.dataset == "health":
        for client in clients_name:
            X = np.array(train_data[client]["x"]).astype(np.float32)

            Y = np.array(train_data[client]["y"]).astype(np.float32)

            if args.sensitive_attr == "race":
                A = train_data[client]["race"]
                args.n_feats = X.shape[1]
            elif args.sensitive_attr == "sex":
                A = train_data[client]["isfemale"]
                args.n_feats = X.shape[1]
            else:
                A = train_data[client]["isfemale"]
                args.n_feats = X.shape[1]
            dataset = Federated_Dataset(X, Y, A)
            client_train_loads.append(DataLoader(dataset, X.shape[0],
                                                 shuffle=args.shuffle,
                                                 num_workers=args.num_workers,
                                                 pin_memory=True,
                                                 drop_last=args.drop_last))

        for client in clients_name:
            X = np.array(test_data[client]["x"]).astype(np.float32)
            Y = np.array(test_data[client]["y"]).astype(np.float32)
            if args.sensitive_attr == "race":
                A = test_data[client]["race"]
            elif args.sensitive_attr == "sex":
                A = test_data[client]["isfemale"]
            else:
                A = np.zeros(X.shape[0])

            dataset = Federated_Dataset(X, Y, A)

            client_test_loads.append(DataLoader(dataset, X.shape[0],
                                                shuffle=args.shuffle,
                                                num_workers=args.num_workers,
                                                pin_memory=True,
                                                drop_last=args.drop_last))

    return client_train_loads, client_test_loads

