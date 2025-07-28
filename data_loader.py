import numpy as np

def load_data(data_dir='dataset', expand_dims=True):
    x_train = np.load(f'{data_dir}/X_train.npy').astype('float32')
    x_test = np.load(f'{data_dir}/X_test.npy').astype('float32')
    x_val = np.load(f'{data_dir}/X_val.npy').astype('float32')

    y_train = np.load(f'{data_dir}/y_train.npy')
    y_test = np.load(f'{data_dir}/y_test.npy')
    y_val = np.load(f'{data_dir}/y_val.npy')

    if expand_dims:
        x_train = np.expand_dims(x_train, axis=-1)
        y_train = np.expand_dims(y_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)
        y_test = np.expand_dims(y_test, axis=-1)
        x_val = np.expand_dims(x_val, axis=-1)
        y_val = np.expand_dims(y_val, axis=-1)

    return (x_train, y_train), (x_test, y_test), (x_val, y_val)