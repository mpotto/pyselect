import os

import requests
import numpy as np
import pandas as pd
from tqdm import tqdm


def get_folder(folder_path, verbose=True):
    """Creates folder, if it doesn't exist, and returns folder path.
    Args:
        folder_path (str): Folder path, either existing or to be created.
    Returns:
        str: folder path.
    """
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        if verbose:
            print(f"-created directory: {folder_path}")
    return folder_path


def download_from_url(file_url, file_path):
    """
    Downloads files from given url. If file has been partially downloaded it will
    continue the download. Requests are sent in chunks of 1024 bytes. Taken from
    https://gist.github.com/wy193777/0e2a4932e81afc6aa4c8f7a2984f34e2.
    Args:
        file_url (str): string with url to download from.
        file_path (str): file path to save downloaded file to.
    Returns:
        int: file size.
    """
    file_size = int(requests.head(file_url).headers["Content-Length"])
    if os.path.exists(file_path):
        first_byte = os.path.getsize(file_path)
    else:
        first_byte = 0

    retries = 0
    while first_byte < file_size:
        if retries > 0:
            print(f"Current number of retries: {retries}")
        retries += 1
        header = {"Range": "bytes=%s-%s" % (first_byte, file_size)}
        pbar = tqdm(
            total=file_size,
            initial=first_byte,
            unit="B",
            unit_scale=True,
            desc=file_url.split("/")[-1],
        )
        req = requests.get(file_url, headers=header, stream=True)
        with (open(file_path, "ab")) as f:
            for chunk in req.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(1024)
        pbar.close()
        first_byte = os.path.getsize(file_path)
    print(f"-saved: {file_path}")
    return file_size


def simple_download_from_url(file_url, file_path):
    """
    Alternative function to download files from url, without support for partial
    downloads and chunking. Useful for some websites that block such functionalities
    or file formats that do not allow partial snapshots.
    Args:
        file_url (str): string with url to download from.
        file_path (str): file path to save downloaded file to.
    """
    print(file_url)
    r = requests.get(file_url, allow_redirects=True)
    open(file_path, "wb").write(r.content)


def get_data(
    data_id,
    seed=0,
):
    """Returns data given data_id identifier. For example, it might return the result
    of splits of the labeled data into train or test sets, or splits into train, val,
    ens and test sets.
    Args:
        data_id (str): Identifier for dataset, in the format dataset_name, dataset_type,
            dataset_fold (e.g., diabetes_full_train, diabetes_full_test;
            diabetes_std_train, diabetes_std_train+ens, diabetes_std_val+ens,
            diabetes_std_ens, diabetes_std_test).
        seed (int): Seed for selecting the train, val, ens and test subsets of dataset.
    Returns:
        pd.DataFrame: Subset of labeled data specified according to data identifier.
    Raises:
        AssertionError: If datatype or dataset provided via data_id is unrecognized; in
            that case there is no proper specification for how to subset labeled data.
    """

    dataset_name, dataset_type, dataset_fold = data_id.split("_")
    assert dataset_type in ["std", "full"]
    assert set(dataset_fold.split("+")).issubset(["train", "ens", "val", "test"])

    # data = dataset_read(dataset_path)
    data = pd.read_csv(f"data/processed/{dataset_name}.csv")

    if dataset_type == "full":
        train_proportion = 0.75

        train_data = data.sample(frac=train_proportion, random_state=seed)
        test_data = data.drop(train_data.index)

        split_data = {
            "train": train_data,
            "test": test_data,
        }
    elif dataset_type == "std":
        train_proportion = 0.6
        test_proportion = 0.25

        test_data = data.drop(
            data.sample(frac=1 - test_proportion, random_state=seed).index
        )
        train_data = data.drop(test_data.index).sample(
            frac=train_proportion, random_state=seed
        )
        val_data = data.drop(train_data.index.union(test_data.index)).sample(
            frac=0.5, random_state=seed
        )
        ens_data = data.drop(
            train_data.index.union(test_data.index).union(val_data.index)
        )

        split_data = {
            "train": train_data,
            "val": val_data,
            "test": test_data,
            "ens": ens_data,
        }

    return pd.concat(split_data[x] for x in dataset_fold.split("+"))


def get_X_y(data_id, seed=0, subset_index=None):
    """Given dataset id and features, returns a random subset
    Args:
        data_id (str): Identifier for dataset, in the format dataset_name, dataset_type,
            dataset_fold (e.g., diabetes_full_train, diabetes_full_test;
            diabetes_std_train, diabetes_std_train+ens, diabetes_std_val+ens,
            diabetes_std_ens, diabetes_std_test).
        seed (int): Seed for selecting the train, val, ens and test subsets of dataset.
        subset_index (np.array): Array with fixed indices to use in subsetting data.
    Returns:
        tuple: feature matrix X and label vector y, both Numpy arrays.
    """
    data = get_data(data_id, seed=seed)
    data = data.fillna(0) + 0  # Fill NA and convert booleans to integers.
    if subset_index is not None:
        data = data.iloc[subset_index]
    X = data.drop(["target"], axis=1)
    y = np.ravel(data[["target"]])
    X = X.to_numpy()
    return X, y
