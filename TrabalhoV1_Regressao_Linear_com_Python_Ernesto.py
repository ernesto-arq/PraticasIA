
# coding: utf-8

# ![title](http://portal.fbuni.edu.br/images/FB-UNI-colorida-300px.png)
# 
# # Trabalho de Inteligência Artificial
# 
# * **Prof. Cleilton Lima Rocha**
# * **email:** climarocha@gmail.com
# * **deadline: 13/04 às 23:59h**
# 
# Para este projeto exploraremos os dados USA_Housing disponível disponível na pasta.
# 
# Para facilitar a vida dos corretores de imóvel você irá realizar a predição de novos imóveis com base no histórico existente. O objetivo do nosso projeto é criar um modelo de aprendizagem supervisionada de regressão e fazer a interpretação dos resultados considerando as métricas de avaliação, além de realizarmos novas descobertas sobre os dados.
# 
# Boa trabalho e hands on!
# 
# **PS.:**
# * Se houver indícios de cola os alunos poderão ter o seu trabalho zerado.
# * O trabalho poderá ser realizado por no máximo 2 pessoas.
# * Quando houver necessidade de splitar os dados aplique a proporção 70 para treino e 30 para teste
# * Quando houver necessidade de utilizar o random_state defina o valor 100
# * O título do email deve ser "Trabalho de IA - Turma 2020.1  - [Membros da equipe]"
# * Envie o código fonte e o report (File ==> Download As ==> Html), com o nome dos membros da equipe, para meu email, climarocha@gmail.com até o dia **13/04 às 23:59h**.

# # Regressão Linear e Polinomial com Python

# #### Realize a importação das bibliotecas necessárias (pandas, numpy, seaborn e sklearn)

# In[1]:


# Instalando Tensorflow
#!pip install -q tensorflow


# In[2]:


# Importando bibliotecas
#import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# #### Crie um dataframe com os dados do arquivo USA_Housing.csv

# In[3]:


df = pd.read_csv('C:/Users/Ernesto/Trabalho_Regressao_Linear/USA_Housing.csv')


# In[4]:


df


# In[5]:


print(df.describe()) 
#Utilizando a função describe para obter dados do data set e ter uma noção dos numeros, valores, medias ete 
#Exemplo media da idade é 29 anos
#Se os dados estao dispersos ou não e etc


# In[6]:


df.head()


# In[7]:


#fazendo primeira verificação de valores nulos e dados e que tipo de dados são apresentados
pd.isnull(df)


# In[8]:


#Verificando se existem valores nulos
#Resultado que não existem dados nulos
df.isnull().values.any()


# In[9]:


#usando comando para retirar outros quaisquer possiveis valores. Embora não existam, 'porem deixar comando salvo'
df.dropna(inplace=True)


# #### Verifique como os dados do preço se distribuem (Dica: Crie um histograma)

# In[10]:


#Confirmar atraves de uma visualização como os dados estão dispostos atraves de uma media de valores "Preços"
# histograma
df.Price.hist(bins = 60)
plt.xlabel("Valor")
plt.ylabel("Quantidade de locais com determinado valor")
plt.title("Distribuição de Valores")
plt.show()


# In[11]:


# Verificando o tipo de corelação que existe entre as variaveis e o nivel de implicação de casualidade
def plot_corr(df, size=20):
    corr = df.corr()    
    fig, ax = plt.subplots(figsize = (size, size))
    ax.matshow(corr)  
    plt.xticks(range(len(corr.columns)), corr.columns) 
    plt.yticks(range(len(corr.columns)), corr.columns)


# In[12]:


#print do grafico gerado
plot_corr(df)


# #### Utilize a correlação para identificar o comportamento das variáveis (Dica: Verifique a função corr()). Quais as descobertas que você conseguiu fazer a partir da análise da correlação?

# In[13]:


#Uma correlação entre os valores em pares de colunas onde os valores eram excluidos valores NA ou seja valores nulos
#Atraves do comando pode-se encontrar a correlação em pares de todas as colunas no quadro de dados imprimido da tela
#Onde
#valores eram excluidos valores NA ou seja valores nulos são excluídos automaticamente. 
#Para qualquer coluna de tipo de dados não numérico no quadro de dados, ela é ignorada.
df.corr()


# #### Se você fosse trabalhar com um modelo linear univariado, qual seria a variável independente que você escolheria como parte do modelo?

# In[14]:


#Resposta:
# Fazendo uma observação ao qual poderia permitir uma análise de cada variável separadamente 
# e refletindo sobre como funciona o sistema economico e usando de conhecimentos sobre 
# os sistema de logistica de imoveis ao qual afetariam os métodos de estatística ao qual determinnam as  variáveis
# seria escolhido sendo X_treino o tamanho de casas o Y_treino o preço
# acredito ser mais relevante na relação
# porem como não existe um tamanho para a casa, mas apenas uma relação de Avg. Area Income	Avg. Area House Age
# que eu não entendi se seria Avg. Area Income "por media do valor do tamanho da casa", 
# usaria então Area media do numero de quartos.


# ## Criando um modelo de regressão linear

# - Como soluções candidatas você pode considerar os seguintes modelos:
#     - [Gradiente descendente estocástico - SGDRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html#sklearn.linear_model.SGDRegressor)
#     - [Regressão linear](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)

# ### Regressão univariada
# Crie um modelo de regressão univariada com base na variávei independente escolhida anteriormente.

# In[15]:


#import tensorflow as tf
#pip install -q tensorflow
# Não consegui importar


# In[16]:


#tratando valores missing ------- inicio do tratamento
# importando preprocessador do pactoe para então dizer os valores faltantes para então serem substituidos
import sklearn as sk


# In[17]:


from sklearn.model_selection import train_test_split


# In[18]:


from sklearn.naive_bayes import GaussianNB


# In[19]:


from sklearn import metrics


# In[20]:


from sklearn.ensemble import RandomForestClassifier


# In[21]:


from sklearn.linear_model import LogisticRegression


# In[22]:


import sklearn 
print (sklearn.__version__)
#Por causa desse problema de versão


# In[23]:


import numpy as np
#from sklearn.impute import SimpleImputer <- codigo abaixo substitui a forma de importação
from sklearn.preprocessing import Imputer


# In[24]:


#Fim das importações


# In[25]:


# Criando objeto para substituição dos valores que faltam pela media, eles serao posterirmente inseridos dessa maneira
# nos dados de treino
preenche = Imputer(missing_values = 0, strategy = "mean")


# In[26]:


# Variável a ser prevista
atrib_prev = ['Avg. Area Number of Rooms']


# In[27]:


# Seleção de variáveis preditoras (Feature Selection)
atributos = ['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', 'Area Population', 'Address']


# In[28]:


# Criando objetos
X = df[atributos].values
Y = df[atrib_prev].values


# In[29]:


#Print dos dados dentro armazenados
print(X)


# In[30]:


#Print dos dados dentro armazenados
print(Y)


# In[31]:


#Fazendo um split, uma divisão de dados para serem treinados


# In[32]:


# Definindo a taxa de split em 30% e 70% como explicados em sala de aula e debatidos
split_test_size = 0.30


# In[33]:


# Criando um modelo de dados para realizar o treino e de teste
X_treino, X_teste, Y_treino, Y_teste = train_test_split(X, Y, test_size=0.30, random_state=101)


# In[34]:


# Imprimindo os resultados de como os dados serão treinados e testados
print("{0:0.2f}% nos dados de treino".format((len(X_treino)/len(df.index)) * 100))
print("{0:0.2f}% nos dados de teste".format((len(X_teste)/len(df.index)) * 100))


# In[35]:


X_treino


# In[36]:


#Confirmação de que não existem valores missing dentro do dataframe que esta sendo desenvolvido e treinado
print("# Linhas no dataframe {0}".format(len(df)))
print("# Linhas missing Avg. Area Income: {0}".format(len(df.loc[df['Avg. Area Income'] == 0])))
print("# Linhas missing Avg. Area House Age: {0}".format(len(df.loc[df['Avg. Area House Age'] == 0])))
print("# Linhas missing Avg. Area Number of Rooms: {0}".format(len(df.loc[df['Avg. Area Number of Rooms'] == 0])))
print("# Linhas missing Avg. Area Number of Bedrooms: {0}".format(len(df.loc[df['Avg. Area Number of Bedrooms'] == 0])))
print("# Linhas missing Area Population: {0}".format(len(df.loc[df['Area Population'] == 0])))
print("# Linhas missing Price: {0}".format(len(df.loc[df['Price'] == 0])))
print("# Linhas missing Address: {0}".format(len(df.loc[df['Address'] == 0])))


# In[37]:


df.head(5)


# In[38]:


#removendo a coluna apenas comentada porque quando ja removida da erro
#removendo o endereço por ser um tipo de dado do tipo string que causara problemas excessivos para ser convertido para tipos
#inteiros!
#Considerado variaveis mais importantes: ---->
# ->>> Renda do lugar, Idade do Imovel, Numero de Quartos, Numero de banheiros, População da Area, Preco do mesmo
df.drop(['Address'],axis=1,inplace=True)


# In[39]:


df.head(5)


# In[40]:


from sklearn.model_selection import train_test_split


# In[41]:


#verificando conjunto total de dados
df


# In[42]:


#PREPARAÇÃO DE DADOS CONCLUIDA!!!!!!!!!!!!! TODOS OS DADOS E AS FORMAS DOS DADOS FORAM CONFERIDOS EM SEUS ESTADOS


# #### Faça a separação dos dados para treinamento
# 
# Vamos criar os nossos dados de treino e teste para o modelo. 
# - **Dica:** Após criar os dados de treino e teste altere a forma para funcionarem corretamente nos modelos, por exemplo X_train = X_train_uni.values.reshape(-1, 1)
# - **Dica:** Defina 30% dos dados para teste

# In[43]:


# removendo uma fature da seleção 'Adress'
atributos = ['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', 'Area Population']


# In[44]:


# reinstanciando os objetos
X = df[atributos].values
Y = df[atrib_prev].values
Y = Y.astype('int')

# reinstanciando modelo de teste
X_treino, X_teste, Y_treino, Y_teste = train_test_split(X, Y, test_size=0.30, random_state = 101)


# In[45]:


# Criando o modelo preditivo
modelo_v1 = GaussianNB()


# In[46]:


# Treinando o modelo
modelo_v1.fit(X_treino, Y_treino.ravel())


# In[47]:


nb_predict_train = modelo_v1.predict(X_treino)


# In[48]:


nb_predict_train


# In[49]:


#Verificando a taxa de acerto do modelo treinado, modelo Naive Bayes

print("Taxa de Acerto: {0:.4f}".format(metrics.accuracy_score(Y_treino, nb_predict_train)))
print()


# In[50]:


nb_predict_test = modelo_v1.predict(X_teste)


# In[51]:


print("Taxa de Acerto: {0:.4f}".format(metrics.accuracy_score(Y_teste, nb_predict_test)))
print()


# In[52]:


#===========================================================================


# In[53]:


#===========================================================================


# In[54]:


#treinando um segundo modelo para analisar os dados baseado em RandomForest para classificação dos dados
modelo_v2 = RandomForestClassifier(random_state = 101)
modelo_v2.fit(X_treino, Y_treino.ravel())


# In[55]:


# Verificando os dados de treino
rf_predict_train = modelo_v2.predict(X_treino)
print("Taxa de Acerto:: {0:.4f}".format(metrics.accuracy_score(Y_treino, rf_predict_train)))


# In[56]:


# Verificando nos dados de teste
rf_predict_test = modelo_v2.predict(X_teste)
print("Taxa de Acerto: {0:.4f}".format(metrics.accuracy_score(Y_teste, rf_predict_test)))
print()


# In[57]:


from sklearn.metrics import classification_report, confusion_matrix


# In[58]:


print("Relatorio de Matrix de Connfusão sobre os dados")
print(classification_report(Y_teste,rf_predict_test))

print("Relatorio de classificação dos dados")
print(classification_report(Y_teste,rf_predict_test))


# In[59]:


#===========================================================================


# In[60]:


#===========================================================================


# In[61]:


from sklearn.linear_model import LogisticRegression


# In[62]:


# Terceira versão do modelo usando fazendo Regressão Logística com os dados
modelo_v3 = LogisticRegression(C = 0.7, random_state = 101, max_iter = 1000)
modelo_v3.fit(X_treino, Y_treino.ravel())
lr_predict_test = modelo_v3.predict(X_teste)


# In[63]:


print("Relatorio de classificação dos dados")
print(classification_report(Y_teste,lr_predict_test))


# In[64]:


### Resultado
## Taxas de acertos do testes
## Usei três modelos para conferir as taxas, sendo o terceiro modelo um modelo de regressão enquanto observava como os dados
#Estavam se comportando e a taxa de acerto entre eles baseado no tipo de conjunto de dados e o 
#tipo de dado que eu havia escolhido
# Em suma ideia provavelmente deveria iniciar uma reformulação na forma que estava tratando a estrutura de dados e escolher
# um novo parametro como novas variaveis alvo apos validar que duvidosamente o modelo Random Forest texe alta taxa de acerto
# Enquanto os modelos anteriores tiveram baixa taxa de Acerto
# Sendo assim deveria reavaliar as entradas e as variaveis alvo

# Modelo usando algoritmo Naive Bayes         = 0.89
# Modelo usando algoritmo Random Forest       = 0.99
# Modelo usando algoritmo Regressão           = 0.30


# ### Regressão multivariada

# In[79]:


#Nova importação dos dados
import pandas as pd 
from datetime import datetime
import numpy as np 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


# In[80]:


#Refazendo data frame
df = pd.read_csv('C:/Users/Ernesto/Trabalho_Regressao_Linear/USA_Housing.csv')


# In[81]:


df.head(5)


# In[82]:


# Verificando novamente a tabela de corelação sendo que o coeficiente segue abaixo
# Coeficiente de correlação: 
# 1  = forte correlação positiva
# 0   = não há correlação
# -1  = forte correlação negativa
df.corr()


# In[83]:


# Definindo as classes
# Numero de banheiros tem as melhores correlações positivas do dataset e menor taxa de correlação negativa
Bedrooms_map = {True : 1, False : 0}


# In[84]:


# Aplicando o mapeamento ao dataset
df['Avg. Area Number of Bedrooms'] = df['Avg. Area Number of Bedrooms'].map(Bedrooms_map)


# In[85]:


# Seleção de novas Features
atributos = ['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Area Population']


# In[86]:


# Executando para os arquivos em data para:  <- Descomentar se necessario executar novamente
#SEXO
local = pd.get_dummies(df['Avg. Area Number of Bedrooms'], prefix='Local', drop_first=True)
df.drop(['Avg. Area Number of Bedrooms'],axis=1,inplace=True)
df = pd.concat([df,local],axis=1)


# In[87]:


# Variável a ser prevista
atrib_prev = ['Price']


# In[89]:


# Criando objetos
X = df[atributos].values
Y = df[atrib_prev].values


# In[90]:


X


# In[91]:


Y


# In[92]:


#verificando valores nulos
df.isnull().values.any()


# In[93]:


# Definindo a taxa de split
split_test_size = 0.30


# In[94]:


# Criando dados de treino e de teste
X_treino, X_teste, Y_treino, Y_teste = train_test_split(X, Y, test_size = split_test_size, random_state = 42)


# In[95]:


lr = LinearRegression()


# In[105]:


#treinando o modelo
modeloMult = lr.fit(X_treino,Y_treino)


# In[108]:


modeloMult


# In[114]:


close_predictions = lr.predict(X_teste)


# In[115]:


from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score
#Se desejar utilizar o RMSE você precisará aplicar sqrt no mean_squared_error de acordo com a fórmula


# In[118]:


mean_squared_error(Y_treino, Y_teste)


# #### Faça a separação dos dados para treinamento com as novas variáveis

# #### Treine o modelo

# #### Analise a performance do modelo, considerando as métricas abaixo

# ## Avaliação do modelo

# #### Analise os coeficientes do modelo
# - O que significa os valores dos coeficientes do modelo?
# - Dica: lm.intercept_

# ### Analisando a qualidade dos resultados

# #### Realize as predições para o modelo univariado
