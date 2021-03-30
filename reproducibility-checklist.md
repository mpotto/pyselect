# The Machine Learning Reproducibility Checklist

## Models and Algorithms
* [ ] A clear description of the mathematical setting, algorithm, and/or model.
* [ ] A clear explanation of any assumptions.
* [ ] An analysis of the complexity (time, space, sample size) of any algorithm.

## Theoretical Claims
* [ ] A clear statement of the claim.
* [ ] A complete proof of the claim. 

## Datasets
* [ ] The relevant statistics, such as number of examples.
* [ ] The details of train / validation / test splits.
* [ ] An explanation of any data that were excluded, and all pre-processing step.
* [ ] A link to a downloadable version of the dataset or simulation environment.
* [ ] For new data collected, a complete description of the data collection process, such as instructions to annotators and methods for quality control.

## Code
* [ ] Specification of dependencies.
* [ ] Training code.
* [ ] Evaluation code
* [ ] (Pre-)trained model(s).
* [ ] `README` file includes table of results accompanied by precise command to run to produce those results.

## Experimental Results
* [ ] The range of hyper-parameters considered, method to select the best hyper-parameter configuration, and specification of all hyper-parameters used to generate results.
* [ ] The exact number of training and evaluation runs.
* [ ] A clear definition of the specific measure or statistics used to report results.
* [ ] A description of results with central tendency (e.g. mean) & variation (e.g. error bars).
* [ ] The average runtime for each result, or estimated energy cost.
* [ ] A description of the computing infrastructure used.