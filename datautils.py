import numpy as np
import arff
from ukf_filter import UKFFilter
import os
from tqdm import tqdm
from args import args
from sklearn.preprocessing import LabelEncoder

def log_msg(message, log_file):
    with open(log_file, 'a') as f:
        print(message, file=f)

def load_UEA(Path='./', folder='Cricket'):

    data_path = Path + folder + '/'

    # train_dataset
    x_train_path = os.path.join(data_path, 'X_train.npy')
    y_train_path = os.path.join(data_path, 'y_train.npy')
    if os.path.exists(x_train_path):
        print(f" '{x_train_path}' already exists, skipping ARFF loading.")
        train_dataset = np.load(x_train_path)
        print(f"Loaded existing X_train.npy: shape {train_dataset.shape}")
        train_label = np.load(y_train_path)
        print(f"Loaded existing y_train.npy: shape {train_label.shape}")
    else:
        train_arff_files = sorted([f for f in os.listdir(data_path) if f.startswith(folder + "Dimension") and f.endswith("_TRAIN.arff")],
                       key=lambda x: int(x.split('Dimension')[1].split('_')[0]))
        if not train_arff_files:
            raise ValueError(f"No ARFF files ending with _TRAIN.arff found in {data_path}")
        train_dataset = []
        for file in train_arff_files:
            with open(os.path.join(data_path, file), "r") as f:
                train_data = arff.load(f)
                train_data_arr = np.array(train_data['data'])
                train_data = train_data_arr[:, :-1].astype(np.float32)
                train_dataset.append(train_data)
        train_dataset = np.array(train_dataset)
        train_dataset = np.transpose(train_dataset, (1, 2, 0))
        print(train_dataset.shape)
        np.save(data_path + 'X_train.npy', train_dataset)
        train_label = np.array(train_data_arr[:, -1]).astype(np.float32)
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(train_label)
        print(integer_encoded.shape)
        np.save(data_path + 'y_train.npy', integer_encoded)

    # test_dataset
    x_test_path = os.path.join(data_path, 'X_test.npy')
    y_test_path = os.path.join(data_path, 'y_test.npy')
    if os.path.exists(x_test_path):
        print(f" '{x_test_path}' already exists, skipping ARFF loading.")
        test_dataset = np.load(x_test_path)
        print(f"Loaded existing X_test.npy: shape {test_dataset.shape}")
        test_label = np.load(y_test_path)
        print(f"Loaded existing y_test.npy: shape {test_label.shape}")
    else:
        test_arff_files = sorted([f for f in os.listdir(data_path) if f.startswith(folder + "Dimension") and f.endswith("_TEST.arff")],
                       key=lambda x: int(x.split('Dimension')[1].split('_')[0]))
        if not test_arff_files:
            raise ValueError(f"No ARFF files ending with _TEST.arff found in {data_path}")
        test_dataset = []
        for file in test_arff_files:
            with open(os.path.join(data_path, file), "r") as f:
                test_data = arff.load(f)
                test_data_arr = np.array(test_data['data'])
                test_data = test_data_arr[:, :-1].astype(np.float32)
                test_dataset.append(test_data)
        test_dataset = np.array(test_dataset)
        test_dataset = np.transpose(test_dataset, (1, 2, 0))
        print(test_dataset.shape)
        np.save(data_path + 'X_test.npy', test_dataset)
        test_label = np.array(test_data_arr[:, -1]).astype(np.float32)
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(test_label)
        print(integer_encoded.shape)
        np.save(data_path + 'y_test.npy', integer_encoded)

    return [train_dataset, train_label], [test_dataset, test_label]

def filter(dataset, filtering = False, filtered_data_saving = True):
    data_path = args.data_path + '/'
    X_filter_path = data_path + 'X_filter.npy'
    if not filtering and os.path.exists(X_filter_path):
        filtered_data_arr = np.load(X_filter_path)
        print(f"Loaded existing X_test.npy: shape {filtered_data_arr.shape}")
    else:
        data_numpy = dataset
        filtered_data_list = []
        for ins in tqdm(range(data_numpy.shape[0]), desc="Processing progress"):
            data = data_numpy[ins]

            # Gets the initial status value
            initial_state = data[0]

            # Instantiate the UKF filter
            ufk = UKFFilter(data=data, initial_state=initial_state, dt=1 / 60)

            filtered_data = ufk.filter()
            filtered_data = np.array(filtered_data)
            filtered_data_list.append(filtered_data)
        filtered_data_arr = np.array(filtered_data_list)
        if filtered_data_saving:
            np.save(X_filter_path, filtered_data_arr)
    return filtered_data_arr

def load_data(Path, folder='boxing'):
    TRAIN_DATA = np.load(Path + '/X_train.npy', allow_pickle=True).astype(np.float)
    TEST_DATA = np.load(Path + '/X_test.npy', allow_pickle=True).astype(np.float)
    TRAIN_LABEL = np.load(Path + '/y_train.npy', allow_pickle=True).astype(np.float)
    TEST_LABEL = np.load(Path + '/y_test.npy', allow_pickle=True).astype(np.float)

    return [TRAIN_DATA, TRAIN_LABEL], [TEST_DATA, TEST_LABEL]

def mean_standardize_transform(X, mean, std):
    return (X - mean) / std

def mean_standardize_fit(X):
    m1 = np.mean(X, axis=1)
    mean = np.mean(m1, axis=0)

    s1 = np.std(X, axis=1)
    std = np.mean(s1, axis=0)

    return mean, std