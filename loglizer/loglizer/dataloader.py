"""
The interface to load OpenStack log datasets. 

"""

import pandas as pd
import os
import numpy as np
import re
from sklearn.utils import shuffle
from collections import OrderedDict

from sklearn.model_selection import train_test_split

def _split_data(x_data, y_data=None, train_ratio=0):
      
    x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.5, random_state=42 )
  
    return (x_train, y_train), (x_test, y_test)

def load_OpenStack(log_file, label_file=None, window='session', train_ratio=0.8, split_type='sequential', random_state=20):
    """ Load OpenStack structured log into train and test data

    Arguments
    ---------
        log_file: str, the file path of structured log.
        label_file: str, the file path of anomaly labels, None for unlabeled data
        window: str, the window options including `session` (default).
        train_ratio: float, the ratio of training data for train/test split.
        split_type: `uniform` or `sequential`, which determines how to split dataset. `uniform` means
            to split positive samples and negative samples equally when setting label_file. `sequential`
            means to split the data sequentially without label_file. That is, the first part is for training,
            while the second part is for testing.
        random_state: int or None, random state for reproducible sampling. Defaults to 20 to match
            the documented benchmark partition (88 train anomalies, 92 test anomalies).

    Returns
    -------
        (x_train, y_train): the training data
        (x_test, y_test): the testing data
        data_df: the processed dataframe
    """

    print('====== Input data summary ======')

    if log_file.endswith('.npz'):
        # Split training and validation set in a class-uniform way
        data = np.load(log_file)
        x_data = data['x_data']
        y_data = data['y_data']
        (x_train, y_train), (x_test, y_test) = _split_data(x_data, y_data, train_ratio)

    elif log_file.endswith('.csv'):
        assert window == 'session', "Only window=session is supported for OpenStack dataset."
        print("Loading", log_file)
        struct_log = pd.read_csv(log_file, engine='c',
                na_filter=False, memory_map=True)
        data_dict = OrderedDict()
        for idx, row in struct_log.iterrows():
            blkId_list = re.findall(r'(instance: \w*-\w*-\w*-\w*-\w*)', row['Content'])
            blkId_set = set(blkId_list)
            for blk_Id in blkId_set:
                if not blk_Id in data_dict:
                    data_dict[blk_Id] = []
                data_dict[blk_Id].append(row['EventId'])
        data_df = pd.DataFrame(list(data_dict.items()), columns=['InstanceId', 'EventSequence'])
        if random_state is not None:
            data_df = data_df.sample(frac=1, random_state=random_state)
        else:
            data_df = data_df.sample(frac=1)
        data_df = data_df.iloc[0:1162]
        
        if label_file:
            # Split training and validation set in a class-uniform way
            label_data = pd.read_csv(label_file, engine='c', na_filter=False, memory_map=True)
            label_data = label_data.set_index('BlockId')
            label_dict = label_data['Label'].to_dict()
            data_df['Label'] = data_df['InstanceId'].apply(lambda x: 1 if label_dict[x] == 'Anomaly' else 0)

            # Split train and test data
            (x_train, y_train), (x_test, y_test) = _split_data(data_df['EventSequence'].values, 
                data_df['Label'].values, train_ratio)
        
            print(y_train.sum(), y_test.sum())       
        
    else:
        raise NotImplementedError('load_OpenStack() only support csv and npz files!')

    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    num_total = num_train + num_test
    num_train_pos = sum(y_train)
    num_test_pos = sum(y_test)
    num_pos = num_train_pos + num_test_pos

    print('Total: {} instances, {} anomaly, {} normal' \
          .format(num_total, num_pos, num_total - num_pos))
    print('Train: {} instances, {} anomaly, {} normal' \
          .format(num_train, num_train_pos, num_train - num_train_pos))
    print('Test: {} instances, {} anomaly, {} normal\n' \
          .format(num_test, num_test_pos, num_test - num_test_pos))

    return (x_train, y_train), (x_test, y_test) , data_df
