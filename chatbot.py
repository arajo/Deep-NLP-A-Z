# Building a ChatBot with Deep NLP


# Importing the libraries
import numpy as np
import tensorflow as tf
import re
import time


################ PART 1 = DATA PREPROCESSING ################


# Importing the dataset
lines = open("data/movie_lines.txt", encoding="utf-8", errors="ignore").read().split("\n")
conversations = open("data/movie_conversations.txt", encoding="utf-8", errors="ignore").read().split("\n")

# Creating a dictionary that maps each line and its id
