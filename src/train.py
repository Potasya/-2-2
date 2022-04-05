#!/usr/bin/env python
# coding: utf-8

# <a id='Q0'></a>
# <center><a target="_blank" href="http://www.propulsion.academy"><img src="https://drive.google.com/uc?id=1McNxpNrSwfqu1w-QtlOmPSmfULvkkMQV" width="200" style="background:none; border:none; box-shadow:none;" /></a> </center>
# <center> <h4 style="color:#303030"> Python for Data Science, Homework, template: </h4> </center>
# <center> <h1 style="color:#303030">Predict the quality of red wine from its physico-chemical properties</h1> </center>
# <p style="margin-bottom:1cm;"></p>
# <center style="color:#303030"><h4>Propulsion Academy, 2021</h4></center>
# <p style="margin-bottom:1cm;"></p>
# 
# <div style="background:#EEEDF5;border-top:0.1cm solid #EF475B;border-bottom:0.1cm solid #EF475B;">
#     <div style="margin-left: 0.5cm;margin-top: 0.5cm;margin-bottom: 0.5cm">
#         <p><strong>Goal:</strong> Practice Linear Regression on wine data</p>
#         <strong> Sections:</strong>
#         <a id="P0" name="P0"></a>
#         <ol>
#             <li> <a style="color:#303030" href="#SU">Set Up </a> </li>
#             <li> <a style="color:#303030" href="#P1">Exploratory Data Analysis</a></li>
#             <li> <a style="color:#303030" href="#P2">Modeling</a></li>
#         </ol>
#         <strong>Topics Trained:</strong> Clustering.
#     </div>
# </div>
# 
# <nav style="text-align:right"><strong>
#         <a style="color:#00BAE5" href="https://monolith.propulsion-home.ch/backend/api/momentum/materials/intro-2-ds-materials/" title="momentum"> SIT Introduction to Data Science</a>|
#         <a style="color:#00BAE5" href="https://monolith.propulsion-home.ch/backend/api/momentum/materials/intro-2-ds-materials/weeks/week2/day1/index.html" title="momentum">Week 2 Day 1, Applied Machine Learning</a>|
#         <a style="color:#00BAE5" href="https://colab.research.google.com/drive/1DK68oHRR2-5IiZ2SG7OTS2cCFSe-RpeE?usp=sharing" title="momentum"> Assignment, Wine Quality Prediction</a>
# </strong></nav>

#    * simplify your notebook to only keep the train-test split, the auto-sklearn model , evaluation and model explanability. __done__
# 
#    * save all ouput figures to images, instead of printing them in the notebook. __done__
# 
#    * implement logging and log the most important steps __done__
# 
#    * save your final model using the dump function of the joblib library, __done__
# 
#    * create a cookiecutter data science project directory in your google drive and track its evolution using git, __done__
# 
#    * place your raw data and your machine learning notebooks in the dedicated folders __done__
# 
#    * convert your notebook into a python script and place it in the dedicated folder __done__
# 
#    * modify the python script and execute it from the command line (use !python myscript.py in another notebook) __done__
# 
#    * create a github account, and a repository where you push your local repository
# 
#    * Use your saved model to build a Flask API with only one “POST” endpoint that returns your prediction when the user posts a list of features

# <a id='SU' name="SU"></a>
# ## [Set up](#P0)

# In[ ]:

#
#get_ipython().system('sudo apt-get install build-essential swig')
#get_ipython().system('curl https://raw.githubusercontent.com/automl/auto-sklearn/master/requirements.txt | xargs -n 1 -L 1 pip install')
#get_ipython().system('pip install auto-sklearn')
#get_ipython().system('pip install -U scikit-learn')
#
#
## In[ ]:
#
#
#get_ipython().system('pip install pipelineprofiler')
#
#
## In[ ]:
#
#
#get_ipython().system('pip install shap')
#get_ipython().system('pip install plotly')


# _Your Comments here_

# In[ ]:


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from sklearn import metrics
from sklearn import set_config
from sklearn.pipeline import Pipeline
from pandas_profiling import ProfileReport
from sklearn.impute import SimpleImputer

from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin

import plotly.express as px
import plotly.graph_objects as go

import autosklearn.regression
import PipelineProfiler
from joblib import dump

import shap

import datetime

import logging


# Please Download the data from [this source](https://drive.google.com/file/d/1gncbcW3ow8vDz_eyrvgDYwiMgNrsgwzz/view?usp=sharing), and upload it on your Introduction2DS/data google drive folder.

## In[ ]:
#
#
#from google.colab import drive
#drive.mount('/content/drive', force_remount=True)


# In[ ]:


data_path = "/content/drive/MyDrive/Introduction2DataScience/tutorials/w2d2/data/raw/"
model_path = "/content/drive/MyDrive/Introduction2DataScience/tutorials/w2d2/models/"
timesstr = str(datetime.datetime.now()).replace(' ', '_')
set_config(display='diagram')


# In[ ]:


log_config = {
    "version":1,
    "root":{
        "handlers" : ["console"],
        "level": "DEBUG"
    },
    "handlers":{
        "console":{
            "formatter": "std_out",
            "class": "logging.StreamHandler",
            "level": "DEBUG"
        }
    },
    "formatters":{
        "std_out": {
            "format": "%(asctime)s : %(levelname)s : %(module)s : %(funcName)s : %(lineno)d : (Process Details : (%(process)d, %(processName)s), Thread Details : (%(thread)d, %(threadName)s))\nLog : %(message)s",
            "datefmt":"%d-%m-%Y %I:%M:%S"
        }
    },
}
logging.config.dictConfig(log_config)


# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# <a id='P1' name="P1"></a>
# ## [Exploratory Data Analysis](#P0)
# 

# ### Understand the Context

# **What type of problem are we trying to solve?**
# 
# With this data set, we want to build a model that would predict the quality of a wine from its physico-chemical characteristics. 
# 
# **_This can be treated either as a classification or a regression problem_**

# **How was the data collected?/ Is there documentation on the Data?**
# 
# Red wine dataset description: 
# 
# **Context**:
# 
# The acidity, alcohol content, as well as other components contents have been measured on wine samples and are reported along with the quality of said wine. the question is: how well can we predict the quality of a wine from these measurements?
# 
# 
# **Content**:  
# 
# For more information, read [Cortez et al., 2009].
# Input variables (based on physicochemical tests):
# 
# 1. fixed acidity
# 2. volatile acidity
# 3. citric acid
# 4. residual sugar
# 5. chlorides
# 6. free sulfur dioxide
# 7. total sulfur dioxide
# 8. density
# 9. pH
# 10. sulphates
# 11. alcohol
# 
# Output variable (based on sensory data):
# 
# 12. quality (score between 0 and 10)

# **Do we have assumption about the data?**

# Most likely, the data will be of different dimensions, which will lead to the fact that sensitive models will be poorly trained on them

# **Can we foresee any challenge related to this data set?**

# They should be scaled so that models can work with them efficiently

# ### Data Structure and types

# **Load the csv file as a DataFrame using Pandas**

# In[ ]:


df = pd.read_csv(f'{data_path}winequality-red.csv', sep=';')


# **How many columns and rows do we have?**

# In[ ]:


df.shape


# **What are the names and meaning of each columns?**

# In[ ]:


df.columns


# **Print the first 10 rows of the dataframe**

# In[ ]:


df.head(10)


# **What are the types of each column?**

# In[ ]:


df.dtypes


# **Do the types correspond to what you expected?
# if not, which columns would you change and why?**

# `quality` type is `int64`, but it's a regression problem so we can say that quality should be a continuous value, means `float64`

# **Perform the necessary type transformations**

# In[ ]:


df = df.astype({'quality': float})
df.dtypes


# Now data is ready to be analysed

# **What are the possible categories for categorical columns?/What is the min, max and mean of each numerical columns?**

# There are no categorical columns. 'Quality' is assumed to be numerical in regression problem

# In[ ]:


df.describe()


# **Perform test/train split here**
# 
# !!! Please think about it!!! How should the data be splitted?

# In[ ]:


test_size = 0.2
random_state = 0


# In[ ]:


train, test = train_test_split(df, test_size=test_size, random_state=random_state)
train.to_csv(f'{data_path}winequality-red-train.csv', index=False)
train= train.copy()
test.to_csv(f'{data_path}winequality-red-test.csv', index=False)
test= test.copy()


# In[ ]:


logging.info(f'train test split with test_size={test_size} and random state={random_state}')


# In[ ]:


print((train.shape[0] - train.count()).to_string())


# ### Missing Values and Duplicates

# **Are there some duplicate columns? rows?**

# In[ ]:


df.columns


# Unfortunately, we do not have such knowledge of chemistry to judge for sure, but I think there are no duplicated columns.

# In[ ]:


train.duplicated(keep=False).value_counts()


# In[ ]:


test.duplicated(keep=False).value_counts()


# There are several duplicates

# **Should we drop duplicate rows?**

# In[ ]:


train = train.drop_duplicates()
test = test.drop_duplicates()


# [Notation for this function](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)

# <a id='P2' name="P2"></a>
# ## [Modelling](#P0)

# In[ ]:


x_train, y_train = train.iloc[:,:-1], train.iloc[:,-1] 


# ### Pipeline Definition

# In[ ]:


time_left=600
time_limit=30


# In[ ]:


autoSklRegressor = autosklearn.regression.AutoSklearnRegressor(
    time_left_for_this_task=time_left,
    per_run_time_limit=time_limit,
)
automl =  autoSklRegressor


# In[ ]:


logging.info(f'Ran autosklearn regressor for a total time of {time_left} seconds, with a maximum of {time_limit} seconds per model run')


# ### Model Training

# In[ ]:


automl.fit(x_train, y_train)


# In[ ]:


profiler_data = PipelineProfiler.import_autosklearn(automl)
PipelineProfiler.plot_pipeline_matrix(profiler_data)


# In[ ]:


dump(automl, f'{model_path}model{timesstr}.pkl')


# In[ ]:


logging.info(f'Saved regressor model at {model_path}model{timesstr}.pkl ')


# ### Model Evaluation

# In[ ]:


x_test, y_test = test.iloc[:,:-1], test.iloc[:,-1]


# In[ ]:


automl.score(x_test, y_test)


# In[ ]:


y_pred = automl.predict(x_test)


# In[ ]:


new_df = pd.DataFrame(np.concatenate((x_test, y_test.to_numpy().reshape(-1,1), y_pred.reshape(-1,1)),  axis=1))
new_df.columns = ['fixed acidity', 'volatile acidity', 'citric acid', 
                  'residual sugar', 'chlorides', 'free sulfur dioxide',
                  'total sulfur dioxide', 'density', 'pH', 'sulphates', 
                  'alcohol', 'True Target', 'Predicted Target']
new_df
list(new_df.iloc[0])


# In[ ]:


mean_squared_error(y_test, y_pred)


# In[ ]:


column = 'fixed acidity'
fig = px.scatter(new_df, x='fixed acidity', y='True Target')
fig.add_trace(go.Scatter(x=new_df['fixed acidity'], y=np.rint(new_df['Predicted Target']),
                    mode='markers',
                    name='model'))
fig.show()


# In[ ]:


logging.info(f"Shapley example saved as {model_path}fixed_acidity_example_{timesstr}.png")


# In[ ]:


from sklearn.metrics import accuracy_score
accuracy_score(new_df['True Target'], np.rint(new_df['Predicted Target']))

