# AutoML with Ludwig
**Duke AIPI 561 MLOPS Project 2 by Shrey Gupta**

## Project Description
This project allows you to train an emotion classification model using the Ludwig AutoML. 

Ludwig is a declarative machine learning framework that allows you to train models and use them for prediction without writing code. It provides a simple, visual way to train models on any type of data. Ludwig is unique in its ability to help make deep learning easier to understand for non-experts and enable faster model improvement iteration cycles for experienced machine learning developers and researchers alike. By using Ludwig, you can quickly train a model without having to write code, and then use the trained model for prediction without writing code. Ludwig’s visualization tool allows you to compare the performance of different models and find the best model for your use case.

In this project, a model is trained using Ludwig AutoML on the [DAIR-AI/emotion](https://huggingface.co/datasets/dair-ai/emotion) dataset of English Twitter messages with six basic emotions: anger, fear, joy, love, sadness, and surprise. Each instance of the dataset looks as follows:
```
{
    "text": "im feeling quite sad and sorry for myself but ill snap out of it soon",
    "label": 0
}
```

## Setting up the project
**1. The project can be run using the following command:**  
```
make install
```
This upgrades pip, installs the requirements and sets up the environment.

**2. To run the formatting on the code:**  
```
make format
```
**3. To run linting on the code:**  
```
make lint
```  
**4. To run all the steps including setup, code formating using black and linting:**  
```
make all
```

## Usage
**1. To train the model:**  
```
python train.py --data-file-path <path_to_data_file> --config-file-path <path_to_config_file>
```

**2. To make predictions using the trained model:**  
```
python predict.py --data-file-path <path_to_data_file> --config-file-path <path_to_config_file> --model-dir <path_to_model_dir>
```

## Project Structure
The project data and code files are arranged in the following manner:

```
├── config                            <- directory for Ludwig model training config files   
    ├── emotion.yaml                  <- yaml config file for emotion classification model
├── data                              <- directory for data files
    ├── data.jsonl                    <- full data file for emotion classification model
    ├── train.jsonl                   <- train data file for emotion classification model
    ├── test.jsonl                    <- test data file for emotion classification model
├── .gitattributes                    <- git attributes file
├── .gitignore                        <- git ignore file
├── LICENSE                           <- license file
├── predict.py                        <- script to make predictions using the trained model
├── Makefile                          <- makefile to run the setup, formatting and linting
├── README.md                         <- description of project and how to set up and run it
├── requirements.txt                  <- requirements file to document dependencies
├── setup.py                          <- setup file for the package
├── train.py                          <- script to train the model
```