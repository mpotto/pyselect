import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

from pyselect.utils import get_folder

data = pd.read_csv("data/processed/amazon.csv")

X = data.drop(["target"], axis=1).to_numpy().flatten()
y = np.ravel(data[["target"]])

# Split into train (0.6), val (0.2), test (0.2)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X,
    y,
    stratify=y,
    train_size=0.8,
    random_state=0,
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    stratify=y_train_val,
    train_size=0.75,
    random_state=0,
)

tfidf_vect = TfidfVectorizer(max_features=2000)

X_train = tfidf_vect.fit_transform(X_train).toarray()
X_val = tfidf_vect.transform(X_val).toarray()
X_test = tfidf_vect.transform(X_test).toarray()
features = tfidf_vect.get_feature_names_out()

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

split_folder = get_folder("data/splitted/amazon")
print("-created directory data/splitted/amazon/")

np.save(f"{split_folder}/X_train.npy", X_train)
np.save(f"{split_folder}/X_val.npy", X_val)
np.save(f"{split_folder}/X_test.npy", X_test)
np.save(f"{split_folder}/y_train.npy", y_train)
np.save(f"{split_folder}/y_val.npy", y_val)
np.save(f"{split_folder}/y_test.npy", y_test)
np.save(f"{split_folder}/features.npy", features)
