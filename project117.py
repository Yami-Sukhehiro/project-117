from google.colab import files
data_to_upload = files.upload()
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

df = pd.read_csv("bank.csv")
factors = df[['variance','skewness','curtosis','entropy']]
X = df["class"]
factors_train, factors_test, X_train, X_test = train_test_split(factors, X, test_size = 0.25, random_state = 0)
scx = StandardScaler()
factors_train = scx.fit_transform(factors_train)
factors_test = scx.transform(factors_test)
classifier = LogisticRegression(random_state = 0)
classifier.fit(factors_train, X_train)
X_predicson = classifier.predict(factors_test)
XPredict = []
for i in X_predicson:
   if i==0:
     XPredict.append("Authorized")

   else:
     XPredict.append("Forged")

XActual = []
for j in X_test.ravel():
  if j==0:
    XActual.append("Authorized")
  else:
    XActual.append("Forged")

labels = ["Forged", "Authorized"]
confumat = confusion_matrix(XActual, XPredict) 
ax = plt.subplot()
sns.heatmap(confumat, annot = True, ax= ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual") 
ax.xaxis.set_ticklabels(labels)
ax.yaxis.set_ticklabels(labels)
ax.set_title("Confusing death")
accuracy = 340/(347)
print(accuracy)