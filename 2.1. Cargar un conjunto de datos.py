#!/usr/bin/env python
# coding: utf-8

# <h2><font color="#004D7F" size=6>Módulo 2. Análisis de datos</font></h2>
# 
# 
# 
# <h1><font color="#004D7F" size=5>1. Cargar un conjunto de datos</font></h1>
# 
# <br><br>
# <div style="text-align: right">
# <font color="#004D7F" size=3>Manuel Castillo-Cara</font><br>
# <font color="#004D7F" size=3>Machine Learning con Python</font><br>

# ---
# 
# <h2><font color="#004D7F" size=5>Índice</font></h2>
# <a id="indice"></a>
# 
# * [1. Introducción](#section1)
# * [2. Cargar CSV](#section2)
#     * [2.1. Desde la librería standard](#section21)
#     * [2.2. Desde NumPy](#section22)
#     * [2.3. Desde Pandas](#section23)
# * [3. Descripción del conjunto de datos](#section3)
#     * [3.1. Clasificación multiclase: IRIS](#section31)
#     * [3.2. Clasifciación Binaria: Sonar, Mines vs. Rocks](#section32)
#     * [3.3. Regresión: Boston House Price](#section33)
# * [4. Conclusiones](#section4)

# In[2]:


# Permite ajustar la anchura de la parte útil de la libreta (reduce los márgenes)
# from IPython.core.display import display, HTML
# display(HTML("<style>.container{ width:98% }</style>"))


# ---
# 
# <a id="section1"></a>
# # <font color="#004D7F"> 1. Introducción</font>

# En esta primera parte de este tema veremos como cargar un conjunto de datos que esté en formato Tidy y, además, veremos como cargar los conjuntos de datos principales que vamos a trabajar a lo largo del curso

# <div style="text-align: right"> <font size=5>
#     <a href="#indice"><i class="fa fa-arrow-circle-up" aria-hidden="true" style="color:#004D7F"></i></a>
# </font></div>
# 
# ---

# <a id="section2"></a>
# # <font color="#004D7F"> 2. Cargar un CSV</font>

# Debe poder cargar sus datos antes de comenzar su proyecto de aprendizaje automático. El formato más común para los datos de aprendizaje automático son los archivos CSV. Hay varias formas de cargar un archivo CSV en Python:
# * Cargue archivos CSV con la biblioteca estándar de Python.
# * Cargue archivos CSV con NumPy.
# * Cargue archivos CSV con Pandas.

# <a id="section21"></a>
# ## <font color="#004D7F">2.1. Desde la librería estándar</font>

# La API de Python proporciona el módulo CSV y funciones `reader()` que se pueden usar para cargar archivos CSV. Una vez cargado, puede convertir los datos CSV a un array NumPy y usarlos para el aprendizaje automático. Por ejemplo, puede descargar el conjunto de datos de los indios Pima en su directorio local con el nombre de archivo `pima-indians-diabetes.data.csv`. Todos los campos en este conjunto de datos son numéricos y no hay una línea de encabezado. El ejemplo carga un objeto que puede iterar sobre cada fila de datos y puede convertirse fácilmente en un array NumPy. Ejecutar el ejemplo imprime la forma del array

# In[2]:


# Load CSV Using Python Standard Library
import csv
import numpy as np
filename= "data/pima-indians-diabetes.csv"
raw_data=open(filename, "r")
reader=csv.reader(raw_data, delimiter= ",", quoting =csv. QUOTE_NONE)
x=list(reader)
data=np.array(x).astype("float")
print(data)
print(data.shape)


# <a id="section22"></a>
# ## <font color="#004D7F">2.2. Desde NumPy</font>

# Puede cargar sus datos CSV usando NumPy y la función `numpy.loadtxt()}`. Esta función no supone una fila de encabezado y todos los datos tienen el mismo formato. El siguiente ejemplo supone que el archivo `pima-indians-diabetes.data.csv` está en su directorio de trabajo actual. Ejecutar el ejemplo cargará el archivo como `numpy.ndarray` e imprimirá la forma de los datos.

# In[3]:


# Load CSV using NumPy
import numpy as np
filename= "data/pima-indians-diabetes.csv"
raw_data=open(filename, "rb")
data=np.loadtxt(raw_data, delimiter = ",")
print(data)
print(data.shape)


# <a id="section23"></a>
# ## <font color="#004D7F">2.3. Desde Pandas</font>

# Puede cargar sus datos CSV usando Pandas y la función `pandas.read_csv()`. Esta función es muy flexible y es quizás mi enfoque recomendado para cargar sus datos de aprendizaje automático. La función devuelve un pandas. [DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html) que puede comenzar a resumir y trazar de inmediato. El siguiente ejemplo supone que el archivo `pima-indians-diabetes.csv` está en el directorio de trabajo actual. Tenga en cuenta que en este ejemplo especificamos explícitamente los nombres de cada atributo al DataFrame

# In[9]:


# Load CSV using Pandas
import pandas as pd
filename= "data/pima-indians-diabetes.csv"
nombres= ["preg","plas","pres","skin","test","mass","pedi","age","class"]
df=pd.read_csv(filename, names=nombres,delimiter=",")
print(df.shape)
df.head(3)


# También podemos modificar este ejemplo para cargar datos CSV directamente desde una URL

# In[10]:


# load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
dfab=pd.read_csv(url, header=None)
dfab


# <div style="text-align: right"> <font size=5>
#     <a href="#indice"><i class="fa fa-arrow-circle-up" aria-hidden="true" style="color:#004D7F"></i></a>
# </font></div>
# 
# ---

# <a id="section3"></a>
# # <font color="#004D7F"> 3. Descripción de conjuntos de datos</font>

# Muchos conjuntos de datos ya vienen ya por defecto en Python a través de la librería [scikit-learn](https://scikit-learn.org/stable/datasets/index.html), lo que significa que no necesita cargar el paquete explícitamente. Estos conjunto de datos se encuentran dentro del módulo `datasets` y de ahí podrán cargarse de manera muy sencillo cargarlo

# In[16]:


import pandas as pd
filename= "data/housing.csv"
columns=["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT","MEDV"]
df=pd.read_csv(filename,names=columns,sep='\s+') #space separated
print(df.shape)
df.head(3)


# Ahora bien, estos conjuntos de datos nos interesa poder convertirlos a un Dataframe de Pandas para poder trabajar con ellos de manera correcta.

# In[22]:


df.describe()


# <a id="section31"></a>
# ## <font color="#004D7F">3.1. Clasificación multiclase: Iris</font>

# El mejor proyecto, de tamaño pequeño, para comenzar con machine learning es el conjunto de datos de [Iris](https://archive.ics.uci.edu/ml/datasets/iris). Este es un buen conjunto de datos para un primer proyecto porque se entiende muy bien. 
# 
# Recordemos algunas características principales:
#    * Los atributos son numéricos, por lo que debemos averiguar cómo cargar y manejar los datos.
#    * Es un problema de clasificación, que nos permite practicar con quizás un tipo más fácil de algoritmo de aprendizaje supervisado.
#    * Es un problema de clasificación multiclase (multi-nominal) que puede requerir un manejo especializado.
#    * Solo tiene 4 atributos y 150 filas, lo que significa que es pequeño y cabe fácilmente en la memoria principal.
#    * Todos los atributos numéricos están en las mismas unidades y la misma escala no requiere ningún escalado especial o transformaciones para comenzar.

# In[25]:


filename="data/iris.data.csv"
names=["Sepal.length","Sepal.width","Petal.lenght","Petal.width","class"]
df=pd.read_csv(filename,names=names)
df


# In[27]:


df.groupby("class").size()


# <a id="section32"></a>
# ## <font color="#004D7F">3.2. Clasificación Binaria: Sonar, Mines vs. Rocks</font>

# El enfoque de este proyecto será el conjunto de datos [Sonar Mines vs Rocks](https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+(Sonar,+Mines+vs.+Rocks)). El problema es predecir objetos de metal o roca a partir de los datos de retorno de la sonda. Cada patrón es un conjunto de 60 números en el rango de 0.0 a 1.0. Cada número representa la energía dentro de una banda de frecuencia particular, integrada durante un cierto período de tiempo. La etiqueta asociada con cada registro contiene la letra "R" si el objeto es una roca y "M" si es una mina (cilindro de metal). Los números en las etiquetas están en orden creciente de ángulo de aspecto, pero no codifican el ángulo directamente.

# In[29]:


filename="data/sonar.all-data.csv"
df=pd.read_csv(filename,header=None)
df


# <a id="section33"></a>
# ## <font color="#004D7F">3.3. Regresión: Boston House Price</font>

# Para este proyecto, trabajaremos el conjunto de datos _Boston House Price._ Cada registro en la base de datos describe un suburbio de la ciudad de Boston. Los datos se extrajeron del área estadística metropolitana estándar de Boston (SMSA) en 1970. Los atributos se definen de la siguiente manera:
# * CRIM: tasa de criminalidad per cápita por ciudad.
# * ZN: proporción de tierra residencial zonificada para lotes de más de 25000 pies cuadrados.
# * INDUS: proporción de acres de negocios no minoristas por ciudad.
# * CHAS: variable ficticia del río Charles (=1 si el trecho delimita el río; 0 de lo contrario).
# * NOX: concentración de óxidos nítricos (partes por 10 millones).
# * RM: número medio de habitaciones por vivienda.
# * AGE: proporción de unidades ocupadas por el propietario construidas antes de 1940.
# * DIS: distancias ponderadas a cinco centros de empleo de Boston.
# * RAD: índice de accesibilidad a autopistas radiales.
# * TAX: tasa de impuesto a la propiedad de valor total por USD10000. 
# * PTRATIO: proporción alumno-profesor por ciudad.
# * B: $1000(Bk - 0.63)^2$ donde Bk es la proporción de personas de color por ciudad
# * LSTAT: % menor estado de la población.
# * MEDV: valor medio de las viviendas ocupadas por sus propietarios en USD1000.

# In[13]:


get_ipython().run_line_magic('pinfo2', '')


# <div style="text-align: right"> <font size=5>
#     <a href="#indice"><i class="fa fa-arrow-circle-up" aria-hidden="true" style="color:#004D7F"></i></a>
# </font></div>
# 
# ---

# <a id="section4"></a>
# # <font color="#004D7F"> 4. Conclusiones</font>

# Llegados a este punto podemos observar la importancia de cómo tengamos nuestro conjunto de datos. Este aspecto es esencial para poder realizar un buen proyecto de Machine Learning y no tener problemas a la hora de la Fase de Modelado. 
# 
# Para este curso usted deberá trabajar un conjunto de datos que esté en formato Tidy Data. Para ello, nos podemos nutrir de varias páginas web para poder elegir un buen conjunto de datos y empezar a prácticar en el Análisis de Datos que empezaremos en la siguiente sección; también para trabajar a lo largo del curso con el mismo conjunto de datos. Para ello, puede escoger de la plataforma [UCI Machine Learning](https://archive.ics.uci.edu/ml/index.php) que, como hemos visto, es un gran repositorio con una gran cantidad de datasets. También podrá escoger de otras páginas existentes, las cuales el enlace se pondrá en el aula virtual.
# 
# En este sentido, se le pide indagar por las diferentes páginas y escoger un conjunto de datos que crea adecuado para seguir el curso.

# <div style="text-align: right"> <font size=5>
#     <a href="#indice"><i class="fa fa-arrow-circle-up" aria-hidden="true" style="color:#004D7F"></i></a>
# </font></div>
# 
# ---
# 
# <div style="text-align: right"> <font size=6><i class="fa fa-coffee" aria-hidden="true" style="color:#004D7F"></i> </font></div>
