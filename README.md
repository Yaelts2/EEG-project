# EEG-Based Phoneme Classification Using Machine Learning
## Project Overview
This project investigates whether individual phonemes can be decoded from EEG signals recorded during a passive listening task. We developed a machine learning pipeline to classify 11 different phonemes, including both vowels and consonants, directly from brain activity.

The core goal of this project is to determine if short, time-locked EEG segments contain sufficient information for a supervised classifier to predict the heard phoneme above chance level. Our findings show that while fine-grained classification is challenging, there are clear and discernible differences between broader phonetic categories, particularly between vowels and consonants.

### Phonemes Classified
The models were trained to classify the following 11 phonemes:

Consonants: /b/, /d/, /p/, /s/, /t/, /z/

Vowels: /i/, /e/, /a/, /u/, /o/
## Data Source
The project uses a publicly available EEG dataset:
Dataset Name: An open-access EEG dataset for speech decoding: Exploring the role of articulation and coarticulation
Source: OpenNeuro (https://doi.org/10.18112/openneuro.ds006104.v1.0.1)
## Key Findings
11-Class Classification: Achieved 31.8% accuracy, above the chance level of ~9%.

Binary Vowel-Consonant Classification: A simplified task resulted in a notable performance boost, reaching 70.7% accuracy
## Project Structure

- **`Notebook/`**: Jupyter notebooks for analysis and visualization.
  - `analysis.ipynb`: Exploratory analysis of and vizualization of raw and processed data 

  - `Train.py`: Script for predicting multi-class classification of 11 phinemes.
  - `Vowel and consonant.py`: Script for predicting binary classification of vowel vs. consonant
  - `Consonant_only.py`: prediction of reduced classes/phonemes with only consonant letters
  - `Preprocess.py`: a script for extracting and preprocessing data from eeg .edf file based on the events.tsv file
 - **`results/`**: Final results 
   - `Classificatio_report.txt`: Result of the multi-class classification of 11 phinemes
  - `report_vowel_and_consonat`: classification report for binary classification of vowel vs. consonant.
  - `Confusion_matrix.png`:confusion matrix for multiclass classification of all the eleven phonemes
  - `Confusion_matrics_vowel_and consonant.png`: confusion matrix for the binary classificatioin
  - `report_consonant only.txt`: classification report for reduced classes/consonant only
  - `Confusion matrix_consonant only`: confusion matrix for reduced number of classes prediction

