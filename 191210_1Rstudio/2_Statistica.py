#!/usr/bin/env python
# coding: utf-8

# <img src="images/logo-uniba.png" style="width:200px" align="right"> 
# <img src="images/pyt.png" style="width:150px" align="left"> 
# <br />
# <br />
# 
# ## Python Basics 3/12/2019
# Cristina Muschitiello - Infocamere, FAO, CREA
# 

# ____
# 
# # PARTE 2: IL LINGUAGGIO
# 
# # 2.3. La statistica

# ___
# 
# ## INDICE DEGLI ARGOMENTI
#
# * LE LIBRERIE DI PYTHON PER LA STATISTICA
# * DATI DI ESEMPIO:
#    * Iris Data
#    * Alcohol Consumption by cuntry
#    * Titanic Data
# * TABELLE DI FREQUENZA
#    * Tabelle singole per variabili quantitative o categoriali
#    * Tabelle singole per variabili continue
#    * Tabelle doppie
# * ALCUNE RAPPRESENTAZIONI GRAFICHE
#    * Istogramma
#    * Diagramma a barre
#    * Diagramma a linee
#    * Diagramma a Torta
#    * Boxplot
# * MEDIE E VARIABILITA'
#    * Media aritmetica
#    * Altri indicatori di tendenza centrale
#    * Deviazione Standard
#    * Varianza
# * SCATTERPLOT, COVARIANZA E CORRELAZIONE
# ___
# 
# ## LIBRERIE DI PYTHON PER LA STATISTICA
# 
# * Numpy
# * Pandas
# * matplotlib
# * Scipy

# In[49]:


# Import Libraries

import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
import statistics as st


# ## DATI DI ESEMPIO

# ### Iris Data

# <img src="images/irisClass.jpg" style="width:400px" align="left"> 

# <img src="images/iris_with_labels.jpg" style="width:200px" align="left"> 

# In[50]:


irisDf = sns.load_dataset('iris')


# In[51]:


print(irisDf)


# In[52]:


irisDf


# ### Alcohol Consumption by Country
# Dati direttamente saricabili dal web

# In[53]:


alcDf = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2010_alcohol_consumption_by_country.csv')


# In[54]:


alcDf


# ## Titanic data
# 
# Dati da scaricare e salvare sul computer.
# 
# I dati sono [scaricabili qui](https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/problem12.html)

# In[55]:


titDf = pd.read_csv("data/titanic.csv") 


# In[56]:


titDf


# TABELLE DI FREQUENZA
## Tabelle singole per variabili quantitative o categoriali

# In[57]:


# Sesso dei passeggeri del Titanic
## Modo1 (funzione della libreria pandas)
pd.crosstab(index=titDf["Sex"],columns="count") 


# In[58]:


## Modo2 (modulo della libreria di default di python)
titDf.Sex.value_counts()


# In[59]:


# Numero di passeggeri per classe
pd.crosstab(index=titDf["Pclass"],columns="cunt")


# In[60]:


# Specie di iris
irisDf.species.value_counts()


# In[61]:


pd.crosstab(index = irisDf["species"],columns = "count")


# ### Tabelle singole per variabili continue

# In[62]:


# Larghezza delle foglie di iris
pd.cut(irisDf['sepal_width'], bins=4)


# In[63]:


irisDf["classes"] = pd.cut(irisDf['sepal_width'], bins=4)
irisDf.classes.value_counts()


# In[64]:


pd.crosstab(index = irisDf["classes"],columns="count")


# ### Tabelle Doppie

# In[65]:


# Sesso per classe dei passeggeri del Titanic
sex_class = pd.crosstab(index = titDf["Sex"],
            columns = titDf["Pclass"])
survived_class = pd.crosstab(index = titDf["Pclass"],
            columns = titDf["Survived"])


# In[66]:


survived_class


# In[67]:


sex_class


# In[68]:


# Aggiunta delle colonne e righe marginali
survived_class = pd.crosstab(index = titDf["Pclass"],
            columns = titDf["Survived"],margins=True)
survived_class


# In[69]:


# Rinominare righe e colonne
survived_class.columns = ["No","Si","Totale"]
survived_class.index = ["Prima","Seconda","Terza","Totale"]
survived_class


# ## ALCUNE RAPPRESENTAZIONI GRAFICHE: Matplotlib e Pandas

# ### Istogramma

# In[70]:


import matplotlib
import matplotlib.pyplot as plt


# In[71]:


# Istogramma semplice
plt.hist(irisDf["sepal_length"], bins=20)
plt.ylabel('sepal_length')
plt.show()


# In[72]:

# Istogramma con densità (con seaborn)
plt.subplots(figsize=(7,6), dpi=100)
sns.distplot(irisDf["sepal_length"] )
plt.show()

# In[73]:


irisDf.plot.hist(subplots=True, layout=(2,2), figsize=(20, 10), bins=20)


# In[74]:


plt.subplots(figsize=(7,6), dpi=100)
sns.distplot( irisDf.loc[irisDf.species=='setosa', "sepal_length"] , color="dodgerblue", label="Setosa")
sns.distplot( irisDf.loc[irisDf.species=='virginica', "sepal_length"] , color="orange", label="virginica")
sns.distplot( irisDf.loc[irisDf.species=='versicolor', "sepal_length"] , color="deeppink", label="versicolor")

plt.title('Iris Histogram')
plt.legend();

plt.show()
# ### Diagramma a barre

# In[75]:


### Specie di iris
# Verticale
plt.subplots(figsize=(7,6), dpi=100)
irisDf['species'].value_counts().plot(kind='bar')
plt.show()

# In[76]:


### Specie di iris
# Verticale
plt.subplots(figsize=(7,6), dpi=100)
irisDf['species'].value_counts().plot(kind='barh')
plt.show()

# In[77]:


# Primi 10 paesi consumatori di alcohol
pl = alcDf[0:10]["alcohol"].plot(kind="barh")
# pl.set_yticklabels(alcDf[0:10]["location"])
# pl.invert_yaxis()
# pl.plot()


# ### Diagramma a torta

# In[78]:


# Primi 10 paesi consumatori di alcohol
fig, ax = plt.subplots(figsize=(10,10))
ax.pie(alcDf[0:10]["alcohol"], labels=alcDf[0:10]["location"], autopct='%.1f%%')
ax.set_aspect('equal')
plt.show()


# ### Box Plot

# In[79]:


# Boxplot per singolo gruppo
fig, ax = plt.subplots(figsize=(10,10))
sns.boxplot( y=irisDf["sepal_length"] )
plt.show()


# In[80]:


# Una variabile numerica e più gruppi
fig, ax = plt.subplots(figsize=(10,10))
sns.boxplot(x=irisDf["species"], y=irisDf["sepal_length"] )
plt.show()


# 3 - Several numerical variable
# Finally we can study the distribution of several numerical variables, let’s say sepal length and width:

# In[81]:


# Più variabili numeriche
fig, ax = plt.subplots(figsize=(10,10))
sns.boxplot(data=irisDf.iloc[:,0:2])
plt.show()


# ___
# ## MEDIE E VARIABILITA'
# 

# ### Media aritmetica
# 
# $\begin{align*}
# \mu = \sum_{i=1}^N{x_i}
# \end{align*}
# $

# In[82]:


st.mean(alcDf['alcohol'])


# In[83]:


mean = st.mean(alcDf['alcohol'])
print("La media è {}".format(mean))


# In[84]:


mean = st.mean(alcDf['alcohol'])
print("La media è {}".format(round(mean,2)))


# ### Altri indicatoridi tendenza centrale

# In[85]:


# Mediana
st.median(alcDf['alcohol'])


# In[86]:


# Media armonica
st.harmonic_mean(irisDf['sepal_length'])


# In[87]:


# Moda
st.mode(irisDf['sepal_length'])


# In[88]:


# Quantili con pandas
irisDf["sepal_length"].quantile([.25, .5, .75])


# In[89]:


irisDf.quantile([.25, .5, .75], axis = 0)


# ### Deviazione Standard

# In[90]:


st.pstdev(alcDf['alcohol'])


# In[91]:


st_dev = st.pstdev(alcDf['alcohol'])

print("La deviazione standard è {}".format(round(st_dev,2)))


# ### Varianza

# In[92]:


st.pvariance(alcDf['alcohol'])


# ___
# ## SCATTERPLOT, COVARIANZA  E CORRELAZIONE 

# In[93]:


# Correlazione fra larghezza e lunghezza dei petali di iris
fig, ax = plt.subplots(figsize=(10,10))
data1 = irisDf["petal_length"]
data2 = irisDf["petal_width"]
plt.scatter(data1, data2)
plt.show()


# In[94]:


# Covarianza
np.cov(data1,data2)


# In[95]:


### Correazione
# Pearson: valore e pvalue
sp.stats.pearsonr(data1,data2) 


# In[96]:


### Correazione
# Spearman: valore e pvalue
sp.stats.spearmanr(data1,data2) 

