#!/usr/bin/env python
# ESG ML Analysis - Kaggle Setup

# Standard libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import re
import warnings
warnings.filterwarnings('ignore')

# PyTorch and deep learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# FastAI
import fastai
from fastai.text.all import *
from fastai.tabular.all import *

# NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import gensim
from gensim import corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel

# Download NLTK resources (run once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Scikit-learn
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.inspection import permutation_importance

# Time series analysis
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Association rule mining
from mlxtend.frequent_patterns import apriori, association_rules

# BERT and transformers
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification

# Visualization
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as pyo
from plotly.subplots import make_subplots
import matplotlib.cm as cm

# Set PyTorch device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Set random seeds for reproducibility
def set_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
set_seeds()

# Display environment information
print(f"PyTorch version: {torch.__version__}")
print(f"FastAI version: {fastai.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# Helper function to save plots
def save_plot(fig, filename, format='png'):
    if not os.path.exists('plots'):
        os.makedirs('plots')
    plt.savefig(f'plots/{filename}.{format}', bbox_inches='tight', dpi=300)
    plt.close(fig)

# Configure pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.3f}'.format)

print("ESG ML Analysis environment setup complete!") 


# Import the 4 main dataframes from the t111111 Kaggle dataset
company_emission_df = pd.read_csv('/kaggle/input/t111111/company_emissions_merged.csv')
target_df = pd.read_csv('/kaggle/input/t111111/Reduktionsziele Results - 20241212 15_49_29.csv')
program1_df = pd.read_csv('/kaggle/input/t111111/Reduktionsprogramme 1 Results - 20241212 15_45_26.csv')
program2_df = pd.read_csv('/kaggle/input/t111111/Reduktionsprogramme 2 Results - 20241212 15_51_07.csv')

# Display basic info about loaded dataframes
print(f"Company-Emission data: {company_emission_df.shape[0]} companies, {company_emission_df.shape[1]} features")
print(f"Target data: {target_df.shape[0]} targets, {target_df.shape[1]} features")
print(f"Program1 data: {program1_df.shape[0]} records, {program1_df.shape[1]} features")
print(f"Program2 data: {program2_df.shape[0]} records, {program2_df.shape[1]} features")




[nltk_data] Downloading package punkt to /usr/share/nltk_data...
[nltk_data]   Package punkt is already up-to-date!
[nltk_data] Downloading package stopwords to /usr/share/nltk_data...
[nltk_data]   Package stopwords is already up-to-date!
[nltk_data] Downloading package wordnet to /usr/share/nltk_data...
[nltk_data]   Package wordnet is already up-to-date!
Using device: cuda
PyTorch version: 2.5.1+cu121
FastAI version: 2.7.18
CUDA available: True
CUDA device: Tesla P100-PCIE-16GB
ESG ML Analysis environment setup complete!
Company-Emission data: 19276 companies, 38 features
Target data: 58501 targets, 27 features
Program1 data: 2873 records, 12 features
Program2 data: 58485 records, 18 features