
# coding: utf-8

# # Aluno: Ernesto Gurgel Valente Neto
# # Matricula: 1020157
# # Turma I.A 2020
# # Trabalho para nota da V2

# # Exercicio de Clusterização
# 
# 
# Vamos trabalhar com o dataset customers. O conjunto de dados original está [disponível na UCI](https://archive.ics.uci.edu/ml/datasets/Wholesale+customers). No nosso dataset foram removidos os atributos 'Channel' e 'Region' para facilitar a análise, pois o foco é verificarmos as seis categorias de produtos comprados pelos clientes, mas fique a vontade para trabalhar com o dado original. Após a remoção das duas variáveis citadas ficamos com o dataset final que será compostos de seis categorias importantes de produtos: Fresh, Milk, Grocery, Frozen, Detergents_Paper e Delicatessen (Perecíveis, Lacticínios, Secos e Molhados, Congelados, Limpeza/Higiene e Padaria/Frios)

# #### Import as bibliotecas

# In[7]:


# Importando bibliotecas
#import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from pandas import DataFrame
from datetime import datetime


# In[8]:


from sklearn.datasets import make_blobs


# #### Acesse os dados disponíveis em customers.csv

# In[9]:


df = pd.read_csv('C:/Users\Ernesto/Downloads/Trabalho_I.A_Dia_5/Wholesale_Customers_data.csv')


# In[10]:


#criando os arquivos de data e os clusters
# Create Data
data = make_blobs(n_samples=200, n_features=6, 
                           centers=4, cluster_std=1.8,random_state=101)


# In[11]:


#Conferindo os dados iniciais pela head para ver se estão corretos segundo o requesito do exercicio de clusterização
#compostos de seis categorias importantes de produtos: 
# Fresh, Milk, Grocery, Frozen, Detergents_Paper e Delicatessen 
#(Perecíveis, Lacticínios, Secos e Molhados, Congelados, Limpeza/Higiene e Padaria/Frios)
df.head()


# In[6]:


df


# In[13]:


#Utilizando a função describe para obter dados do data set e ter uma noção dos numeros, valores, medias ete 
#Exemplo media da idade é 29 anos
#Se os dados estao dispersos ou não e etc
print(df.describe())


# In[14]:


#fazendo primeira verificação de valores nulos e dados e que tipo de dados são apresentados
pd.isnull(df)


# In[15]:


#usando comando para retirar outros quaisquer possiveis valores. Embora não existam, 'porem deixar comando salvo'
df.dropna(inplace=True)


# In[16]:


#Verificando se existem valores nulos
#Resultado que não existem dados nulos
df.isnull().values.any()


# In[18]:


#Uma correlação entre os valores em pares de colunas onde os valores eram excluidos valores NA ou seja valores nulos
#Atraves do comando pode-se encontrar a correlação em pares de todas as colunas no quadro de dados imprimido da tela
#Onde
#valores eram excluidos valores NA ou seja valores nulos são excluídos automaticamente. 
#Para qualquer coluna de tipo de dados não numérico no quadro de dados, ela é ignorada.
df.corr()


# In[19]:


#removendo a coluna apenas comentada porque quando ja removida da erro
df.drop(['Channel'],axis=1,inplace=True)


# In[20]:


#removendo a coluna apenas comentada porque quando ja removida da erro
df.drop(['Region'],axis=1,inplace=True)


# In[21]:


df.head()


# In[22]:


df


# In[23]:


df.corr()


# ## Visualizando os Dados

# In[24]:


#plt.scatter(df[0][:,0],df[0][:,1],c=df[1],cmap='rainbow')


# In[25]:


#plt.scatter(df[;,0][:,1],c=kmeans_clusters.labels_,camp='rainbow')
#plt.scatter(kmeans_clusters.cluster_center_[:,0],kmeans_clusters.cluster_centers_[:,1],color='black')
#plt.show()


# #### Aplique pré-processamento sobre os dados para criar seu dataset alvo, se você julgar necessário. 
# - Dica: Você pode considerar remover as amostras que são outiliers em mais de um atributo.

# In[33]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


#importando o metodo da simulheta para gerar um plot para imaginar quantos clusters seriam usados
#Incliando melhor numero de casos como 5
from sklearn.metrics import silhouette_score

for k in range(2, 100):
    kmeans_ = KMeans(n_clusters=k, random_state=10)
    kmeans_.fit(data[0])
    print(k, silhouette_score(data[0], kmeans_.predict(data[0])))


# In[ ]:


# Analisando os dados graficos e o numero de clusters pelo metodo da simulheta foi:
# pelos grupos e tipos de grupo e interpretação


# ## Criando os clusters

# #### Aplique o algoritmo [K-means](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html), sobre o dataset criado. 
# - Dicas: 
#     * Julgue a necessidade de aplicar [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) ou não. Veja mais em nota, no final do exercício.
#     * Aplique a [normalização](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html), se for utilizar o PCA
#     * Se for aplicar o PCA utilize o fit_transform() para facilitar sua vida
#     * Para reverter os valores em alguma análise considere reverter os valores utilizando o método inverse_transform() do pipeline criado

# #### Aplique o método da [silhueta](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html#sklearn.metrics.silhouette_score) e verifique a qualidade do cluster gerado
#  - Dicas: No método da silhueta você deve considerar kmeans_.predict(data) como sendo os **labels**

#  #### Varie o número de K, hiper parâmetro do K-Means e identifique o melhor valor considerando o método da silhueta.

#  #### Desafio: 
#    - Varie o número de K, hiper parâmetro do K-Means e identifique o melhor valor considerando o método da curva do cotovelo, [veja este exemplo](https://pythonprogramminglanguage.com/kmeans-elbow-method/). O resultado obtido foi igual ao aplicado na questão anterior?

# #### Como você interpreta os resultados parecem favoráveis os centroídes são de fato distintos? Como você interpreta os dados com base nesta informação? 
# - Dica.: Para recuperar os centroídes use kmeans.cluster_centers_

# #### Desafio: 
#    - Recupere os centroídes e faça um radar chart considerando os centroídes, [veja este exemplo](https://python-graph-gallery.com/391-radar-chart-with-several-individuals/).
#        - Dica.: Para recuperar os centroídes use kmeans.cluster_centers_

# #### Recupere as amostras de cada cluster e faça um parallel coordinates, [veja este exemplo](https://python-graph-gallery.com/150-parallel-plot-with-pandas/) ou [este](https://jovianlin.io/data-visualization-seaborn-part-2/). Os resultados parecem favoráveis e os centroídes são de fato distintos? Como você interpreta os dados com base nesta informação?

# #### Crie alguns gráficos, scatterplot do cluster, mostrando os dados definidos pelo cluster.
# - Que tal tentar marcar os centroídes neste gráfico? *P.S.: Se você criou o PCA construa o gráfico considerando variações das componentes*

# # Nota

# - É muito comum aplicar PCA para reduzir a dimensionalidade dos dados, principalmente quando trabalhamos com aprendizagem não supervisionada. Para isso é necessário interpretar o quanto as componentes explicam a variação nos dados, por exemplo, qual o número de componentes utilizar e o cumulativo total que representa as componentes sobre os dados. Também é preciso explicar como as componenentes se relaciona as variáveis originais do dado, considerando todas as amostras ou parte delas. Nesses dois sites há dicas de como compreender melhor essa interpretação [dica_1](https://support.minitab.com/en-us/minitab/18/help-and-how-to/modeling-statistics/multivariate/how-to/principal-components/interpret-the-results/key-results/) e [dica_2](https://newonlinecourses.science.psu.edu/stat505/node/54/)

# - **Desafio:** Se você utilizou o PCA realize a análise das componentes como valores acumulados e correlação
