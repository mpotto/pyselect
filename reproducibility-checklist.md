# The Machine Learning Reproducibility Checklist (annotated)

**PS** stands for proposed solution.

## Models and Algorithms
* [ ] A clear description of the mathematical setting, algorithm, and/or model.
    * **PS**: well-written project documentation and article.
* [ ] A clear explanation of any assumptions.
    * **PS**: well-written project documentation and article.
* [ ] An analysis of the complexity (time, space, sample size) of any algorithm.
    * **PS**: use experimens logs, save hardware requirements, show memory and computation usage, analyze computational complexity (assymptotic behavior).

## Theoretical Claims
* [ ] A clear statement of the claim.
    * **PS**: well-written article.
* [ ] A complete proof of the claim. 
    * **PS**: well-written article.

## Datasets
* [ ] The relevant statistics, such as number of examples.
    * **PS**: include summary in article, add README.md summary on data folder.
* [ ] The details of train / validation / test splits.
    * **PS**: well written-article, documented experimentes in notebooks. Split notebooks into datasets.
* [ ] An explanation of any data that were excluded, and all pre-processing step.
    * **PS**: Clear notebooks, DVC pipelines and usage.
* [ ] A link to a downloadable version of the dataset or simulation environment.
    * **PS**: Poetry isolation, DVC remote storage.
* [ ] For new data collected, a complete description of the data collection process, such as instructions to annotators and methods for quality control.
    * **PS**: instruction on synthetic data generation. Well-documented `pyselect.synthesizer` submodule.

## Code
* [x] Specification of dependencies.
    * **PS**: Poetry.
* [ ] Training code.
    * **PS**: notebooks.
* [ ] Evaluation code
    * **PS**: notebooks.
* [ ] (Pre-)trained model(s).
    * **PS**: save models as serialized objects available at each experiment. Write instructions on usage.
* [ ] `README` file includes table of results accompanied by precise command to run to produce those results.
    * **PS**: add `README.md` results of an examples folder.

## Experimental Results
* [ ] The range of hyper-parameters considered, method to select the best hyper-parameter configuration, and specification of all hyper-parameters used to generate results.
    * **PS**: Optuna scripts, optuna sampler specs, tables of hyperparameters in article.
* [ ] The exact number of training and evaluation runs.
    **PS**: Notebooks specification (make a summary table). Article information. 
* [ ] A clear definition of the specific measure or statistics used to report results.
    * **PS**: clear definition of loss and evaluation metrics on article.
* [ ] A description of results with central tendency (e.g. mean) & variation (e.g. error bars).
    **PS**: run many experiments (specify how many) and supply those statistics. 
* [ ] The average runtime for each result, or estimated energy cost.
    * **PS**: Ignite time profiling.
* [ ] A description of the computing infrastructure used.
    * **PS**: Write a hardware specification `README.md`-like file.
