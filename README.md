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
Source: OpenNeuro [ds006104] (https://doi.org/10.18112/openneuro.ds006104.v1.0.1)
## Key Findings
11-Class Classification: Achieved 31.8% accuracy, above the chance level of ~9%.

Binary Vowel-Consonant Classification: A simplified task resulted in a notable performance boost, reaching 70.7% accuracy
