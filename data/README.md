# Data

## Naming and Storage standard

### Synthetic Datasets

Every dataset must be stored in a single subfolder of `data/synthetic`. The folder name should follow the structure:

`{Synthesizer function name}-{Number of samples}-{Number of features}-{Number of informative features}`

The datasets should be split in train-test before being consumed by machine learning algorithms. Inside the dataset folder, the train or test data should be named as

`{Synthesizer function name}-{Number of samples}-{Number of features}-{Number of informative features}-{train, test}.pt`

Inside each dataset folder there must be a `README.md` (human-readable file) containing basic information (number of samples, number of features, raw or processed, processing steps and its implementation) of the dataset and its components. Additional files should be listed and described.
