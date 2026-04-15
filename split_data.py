import random
import os
import ast
import pandas as pd


def read_affinity_file(file_path, sep=r'\s{2,}'):
    """ Reads the affinity data from a file, expecting multiple spaces as delimiters"""
    #return pd.read_csv(file_path, sep=r'\s{2,}', header=None, engine='python'
    return pd.read_csv(file_path, sep=sep, header=None, engine='python')



def check_affinity_file(DATA_PATH, use_Data = True, sep=r'\s{2,}', file_name = "pIC50.txt", cols = 2):
    """Creates the Y affinity matrix from an input Affinity.txt file"""
    path_affinity_file = os.path.join(DATA_PATH, file_name)
    df = read_affinity_file(path_affinity_file, sep=sep)
    if df.shape[1] < cols:
        raise ValueError(f"The affinity file must have at least {cols} columns.")
        #Create a transposed affinity matrix with 5 additional NaN columns
   # print("Affinity matrix in correct format")
    if use_Data:
        return df
    else:
        return None


def df_to_dict(df, data_info):
    """Dataframe to dictionary, first column of the dataframe will be the keys and the values will be extracted from
    the data_info column name"""
    if df.empty:
        print("Error: DataFrame is empty.")
        return {}

    key_col = df.columns[0]

    # Case 1: data_info exists as a value in first column
    if data_info in df[key_col].values:
        return dict(zip(df[key_col], df[data_info]))

    # Case 2: first row contains headers (including data_info)
    first_row = df.iloc[0].astype(str).values
    if data_info in first_row:
        # Promote first row to header
        df_fixed = df.copy()
        df_fixed.columns = first_row
        df_fixed = df_fixed.iloc[1:]

        key_col = df_fixed.columns[0]

        if data_info not in df_fixed.columns:
            print(f"Error: '{data_info}' not found in promoted headers.")
            return {}

        return dict(zip(df_fixed[key_col], df_fixed[data_info]))

    print(
        f"Error: '{data_info}' not found in first column values "
        "or in first row headers."
    )
    return {}


def check_leftovers(num_folds, fold_size, sequence):
    left_overs = [False, False]
    total_assigned = fold_size * num_folds
    leftovers = sequence[total_assigned:]
    if len(leftovers) > 0:
        left_overs[0] = True
        if len(leftovers) >= num_folds // 2:
            left_overs[1] = True
    return left_overs, leftovers


def split_train_valid(data, valid_split_ratio, seed):
    """Splits data into training and validation sets."""
    random.seed(seed)
    random.shuffle(data)
    valid_size = int(len(data) * valid_split_ratio)
    return data[valid_size:], data[:valid_size] #train,valid


def create_k_fold_splits(common_keys, num_folds, valid_ratio, seed=43, shuffle=True):
    """
    True K-fold CV:
        - Each fold's test set = exactly 1/num_folds of the dataset
        - Remaining data is split into train and validation using valid_ratio

    Args:
        common_keys (list): dataset items
        num_folds (int): number of folds (K)
        valid_ratio (float): fraction of (train+valid pool) that goes to validation
                             Example: 0.2 means 20% of the non-test data becomes validation
        seed (int): reproducible randomness
        shuffle (bool): shuffle data before splitting

    Returns:
        (train_folds, valid_folds, test_folds)
    """

    # --- Prepare keys ---
    keys = list(common_keys)
    rng = random.Random(seed)

    if shuffle:
        rng.shuffle(keys)

    N = len(keys)

    # --- Build K equal-sized test folds ---
    base, extra = divmod(N, num_folds)
    fold_sizes = [base + (1 if i < extra else 0) for i in range(num_folds)]

    test_folds = []
    start = 0
    for size in fold_sizes:
        test_folds.append(keys[start:start + size])
        start += size

    train_folds = []
    valid_folds = []

    # --- For each fold, train/valid = complement of test ---
    for i in range(num_folds):

        # pool = all data except this fold's test set
        pool = [x for j, fold in enumerate(test_folds) if j != i for x in fold]

        # Shuffle pool with a per-fold seed
        rng_i = random.Random(seed + i + 1000)
        rng_i.shuffle(pool)

        # Validation size is a fraction of pool
        valid_size = int(len(pool) * valid_ratio)

        valid = pool[:valid_size]
        train = pool[valid_size:]

        train_folds.append(train)
        valid_folds.append(valid)

    return train_folds, valid_folds, test_folds





def k_fold_data_split(common_keys, num_folds, valid_split_ratio):
    """
    Perform k-fold splitting, ensuring no element is repeated in the test sets,
    and compute corresponding training sets.

    Args:
        common_keys: list of names.
        train_perc (float): Percentage of data to use for training (e.g., 0.8 for 80%).
        num_folds (int): Number of folds.

    Returns:
        tuple: (list of test sets, list of training sets for each fold).
    """

    random.seed(42)
    random.shuffle(common_keys)

    # Compute k-fold splits
    train_folds, valid_folds, test_folds = create_k_fold_splits(common_keys, num_folds, valid_split_ratio)

    return train_folds, valid_folds, test_folds


def write_fold_to_file(fold_data, file_path):
    """Writes a list (of folds) to a file in Python list format."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as file:
        file.write('[' + ', '.join(map(str, fold_data)) + ']')


def read_fold_from_file(file_path):
    """Reads a list from a file written in Python list format."""
    with open(file_path, 'r') as file:
        content = file.read()
        return ast.literal_eval(content)


def train_valid_test_folds(common_keys, num_folds, valid_split_ratio, folds_folder, train_file_name, valid_file_name, test_file_name,
                           DATA_PATH, DATA_OUT):
    """Creates the train and test files for the dataset"""
    # Write the test and train fold settings to their respective files
    # Y = pickle.load(open(os.path.join(DATA_PATH, "Y"), "rb"), encoding='latin1')
   # affinity_data = check_affinity_file(DATA_PATH, True, sep="  ", file_name = "pIC50.txt")
    folder_file_path = os.path.dirname(DATA_OUT)
    train_file_path = os.path.join(folder_file_path,'split_data', train_file_name)
    valid_file_path = os.path.join(folder_file_path,'split_data', valid_file_name)
    test_file_path = os.path.join(folder_file_path, 'split_data',test_file_name)

    if not os.path.exists(train_file_path) and not os.path.exists(valid_file_path) and not os.path.exists(test_file_path):
        train_folds, valid_folds, test_folds  = k_fold_data_split(common_keys, num_folds, valid_split_ratio)
        write_fold_to_file(train_folds, train_file_path)
        write_fold_to_file(valid_folds, valid_file_path)
        write_fold_to_file(test_folds,  test_file_path)
    else:
        print(f'Files inside {folds_folder} already exists, check if you wish to make any change on it')
        print('Extracting folds index information')
        train_folds = read_fold_from_file(train_file_path)
        valid_folds = read_fold_from_file(valid_file_path)
        test_folds = read_fold_from_file(test_file_path)

    return train_folds, valid_folds, test_folds


    #except Exception as e:
    #    print(f"Error creating the train/test files: {(e)}")