import os
import re

import numpy as np
import pandas as pd
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def process_text(data, n_samples, random_state):
    """Processing of Amazon Fine Food Reviews based on the tutorial
    https://www.kaggle.com/code/shashanksai/text-preprocessing-using-python.
    Dataset comes from http://i.stanford.edu/~julian/pdfs/www13.pdf.
    """
    # Remove "neutral" category.
    data_score_removed = data[data["Score"] != 3]

    # Define new score vector.
    score_update = data_score_removed["Score"]

    # Map scores to sentiment (0 or 1).
    sentiment = score_update.map(lambda x: 0 if x < 3 else 1)

    # Insert sentiment on score column.
    pd.options.mode.chained_assignment = None  # Avoid "chained assignment" warning.
    data_score_removed.loc[:, ["Score"]] = sentiment

    # Drop duplicates where needed.
    data_drop_duplicates = data_score_removed.drop_duplicates(
        subset={"UserId", "ProfileName", "Time", "Text"}
    )

    # Validate entries based on definition of numerator and denominator
    # features.
    final = data_drop_duplicates[
        data_drop_duplicates["HelpfulnessNumerator"]
        <= data_drop_duplicates["HelpfulnessDenominator"]
    ]

    # Subset data.
    if n_samples is not None:
        final = final.sample(n=n_samples, random_state=random_state)

    X_text = final["Text"]
    y = final["Score"].to_numpy()

    sent = []
    snow = SnowballStemmer("english")
    for i in tqdm(range(len(X_text))):
        sentence = X_text.iloc[i]
        sentence = sentence.lower()  # Converting to lowercase
        cleaner = re.compile("<.*?>")  # Define the pattern for HTML tags.
        sentence = re.sub(cleaner, " ", sentence)  # Removing HTML tags
        sentence = re.sub(r'[?|!|\'|"|#]', r"", sentence)  # Removing marks.
        sentence = re.sub(r"[.|,|)|(|\|/]", r" ", sentence)  # Removing punctuations

        sequ = " ".join(
            [
                snow.stem(word)
                for word in sentence.split()
                if word not in stopwords.words("english")
            ]
        )
        sent.append(sequ)

    X_text = np.array(sent)

    return X_text, y


def vectorize_and_scale(X_train, X_test, y_train, y_test):

    tfidf_vect = TfidfVectorizer()

    X_train = tfidf_vect.fit_transform(X_train).toarray()
    X_test = tfidf_vect.transform(X_test).toarray()

    scaling = StandardScaler()
    X_train = scaling.fit_transform(X_train)
    X_test = scaling.transform(X_test)

    features = tfidf_vect.get_feature_names_out()

    return X_train, X_test, y_train, y_test, features


# Open data.
filename = os.path.join("data/amazon/archive", "Reviews.csv")
data = pd.read_csv(filename)
print(f"Data has original shape {data.shape}.")

# Define seed (for pandas sampling reproducibility).
rng = np.random.RandomState(0)

# Process text.
if all(
    list(
        map(
            os.path.isfile,
            ["data/amazon/processed/X_text.npy", "data/amazon/processed/y.npy"],
        )
    )
):
    X_text = np.load("data/amazon/processed/X_text.npy")
    y = np.load("data/amazon/processed/y.npy")
else:
    X_text, y = process_text(data, n_samples=150 * 10 ** 3, random_state=rng)
    np.save("data/amazon/processed/X_text.npy", X_text)
    np.save("data/amazon/processed/y.npy", y)

# Split data.
X_train, X_test, y_train, y_test = train_test_split(
    X_text, y, train_size=0.8, test_size=0.2, random_state=rng
)

# Vectorize and scale.
X_train, X_test, y_train, y_test, features = vectorize_and_scale(
    X_train, X_test, y_train, y_test
)

np.save("data/amazon/processed/X_train.npy", X_train)
np.save("data/amazon/processed/X_test.npy", X_test)
np.save("data/amazon/processed/y_train.npy", y_train)
np.save("data/amazon/processed/y_test.npy", y_test)
np.save("data/amazon/processed/features.npy", features)
