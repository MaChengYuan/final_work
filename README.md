# Development of an Approach to Automation of Foreign Enrollee Intellectual Support Based on NLP-Technologies and Information Crawling

## Table of Contents

1. Project Description
2. Technologies
3. Files Description
4. Running Files
5. Potential development
6. Licensing and Authors

## Project Description
The project aims at develpement of intellectual enrollee system for university side with semi-automatic workflow minimizing interference of man-power. This study mainly focuses on designing enrollment support systems from scratch, which is able to assign responses to given questions with support of current existing start-of-art approaches and NLP model. The whole system is a hybrid NLP works enlightened by multiple NLP studies, from the design of pre-trained language models, data augmentation approaches to sequential recommendation systems. Each approach, there is slight modification applied without changing the core functionality to optimize the capability based on requirement of current work. 

## Technologies
Technique : 
* Hypercomplex classifer implemented in pre-trained RoBERT model
* GPT augmentation with Pos-tagging
* SASrec sequential recommendation system

Platform :
* Telegram

Framework :
* Pytorch

## Main File description

* init.py : file for activation of whole system

* main.py : file for user interface of telegram

* function.py : file for storage of functions linked to main.py

* mongodb_read.py : file containing funcion of CRUD for MONGODB

* retrain.py : activate RoBERT model to train

* recommend.py : contains function to train SASrec sequential prediction model

## Setup

>[!NOTE]
>In this directory, tokens for telebot api, MongoDB and password for admin mode do not contain.
>To activate system, it is neccessary to make file named after password.txt followed format in token.py


