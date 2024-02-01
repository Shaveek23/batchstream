BatchStream Learn library
============
Authors: Paweł Golik
---

The creation of this library stemmed from the research conducted during the pursuit of a Master of Science thesis titled 'Online learning and batch learning methods for data streams' at the Faculty of Mathematics and Information Science at Warsaw University of Technology during the academic year 2022/23.

---

## About

The library's primary objective is to establish a cohesive framework encompassing online learning, batch learning, and a hybrid blend of both techniques for the classification of data streams. This is designed to streamline the process of conducting proof-of-concept research experiments using constrained data streams, offering a straightforward approach for testing various methods. 
While the library currently provides support for [`scikit-learn Pipelines`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) and [`River`](https://riverml.xyz/0.15.0/) classifiers, it has been architected to seamlessly collaborate with any online or batch learning library of your choice. The only requirement is to provide an implementation of a dedicated class that facilitates the integration of third-party library APIs with the batchstream learn API.

---

## Repository structure

Directories:
- `batchstream` - contains a source code of the library. Each subdirectory contains implementation of:
    - `batchstream_monitoring_strategy` - monitoring strategies used by batch models. These objects defines what are `reference` and `current` batches used by the `Evidently AI` to detect concept drift.
    - `combine` - ensembles combining online and batch learning methods.
    - `drift_handlers` - classes responsible for execution of a given drift handling strategy (detection and model adaptation).
    - `estimators` - wrappers integrating third-party libraries (such as `scikit-learn`) with the `BatchStream` library API.
    - `evaluation` - classes performing evaluation and logging of models performance.
    - `experiment` - the main class performing a predefined experiment on a constrained data stream read from a file.
    - `history` - a class responsible for caching historical observations for batch learning models.
    - `model_comparers` - objects comparing two batch learning models (after retraining).
    - `monitoring` - objects providing data and model monitoring. Announce concept drift when detected.
    - `pipelines` - stream learning pipelines that wrap online and batch learning models.
    - `retraining_strategy` - defines a retraining batch of data for batch learning models that needs to adapt to changes.
    - `utils` - functions enabling logging, reading results, visualization and others.
- `data` - contains files with constrained data streams
- `experiments` - contains implementation of functions setting up different experiments
- `scripts` - contains scripts defining performed experiments for employed data streams
- `utils` - contains functions implementing artificial data streams generation, reading data from files etc.
- `demo.py` - contains a simple library usage demo.

Files:
- `requirements.txt` - a file listing all library dependencies, it can be used to recreate a python environment.
- `start.sh` - a bash script running a given file from the `scripts` directory on a HPC cluster.
---

## Environment setup

**1. Install Conda:**
If Anaconda is not already installed on the other computer.

**2. Create the Conda Environment:**
```bash
conda env create -f environment.yml
```

**3. Activate the Conda Environment:**
```bash
conda activate your_environment_name
```

---
## Used libraries

- [**River**](https://riverml.xyz/0.15.0/): is a versatile library designed for constructing online machine learning models.
- [**scikit-learn**](https://scikit-learn.org/stable/getting_started.html): is a library that offers support for both supervised and unsupervised learning. Additionally, it offers a wide array of tools for tasks such as model fitting, data preprocessing, model selection, model evaluation, and various other utilities.
- [**Evidently AI**](https://www.evidentlyai.com/): is an open-source tool that is valuable for evaluating, testing, and monitoring machine learning models and data.
