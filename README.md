# HMS Harmful Brain Activity Classification

This project is part of a Kaggle competition hosted by Harvard Medical School, focused on classifying harmful brain activity using medical data. The goal is to develop a model that accurately predicts harmful brain activity based on provided datasets.

## Table of Contents
- [Getting Started](#getting-started)
- [Dependencies](#dependencies)
- [Data Acquisition](#data-acquisition)
- [Usage](#usage)

## Getting Started

To get started with the project, you need to set up the development environment and install necessary dependencies.

### Dependencies

This project uses `pipenv` for managing Python dependencies. Ensure you have `pipenv` installed on your machine. If not, you can install it using:

```bash
pip install pipenv
```

### Setting Up the Environment

1. Set up the environment by running the following command:

    ```bash
    export PIPENV_VENV_IN_PROJECT=1
    ```

2. Install the project dependencies using `pipenv`:

    ```bash
    pipenv install
    ```

## Data Acquisition

Download the dataset directly from Kaggle using the Kaggle API. If you haven't set up the Kaggle API, follow the instructions [here](https://www.kaggle.com/docs/api) to configure it.

1. Download the dataset:

    ```bash
    kaggle competitions download -c hms-harmful-brain-activity-classification
    ```

2. Unzip the downloaded dataset into the `data` directory:

    ```bash
    unzip hms-harmful-brain-activity-classification.zip -d data
    ```

3. Remove the zip file to save space:

    ```bash
    rm hms-harmful-brain-activity-classification.zip
    ```

## Usage

To train and evaluate the model, follow the instructions provided in the main script or notebook file. Ensure all dependencies are correctly installed, and the data is placed in the correct directories as specified.
