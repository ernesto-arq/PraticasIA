
# coding: utf-8

# ![title](http://portal.fbuni.edu.br/images/FB-UNI-colorida-300px.png)
# 
# # Trabalho de Inteligência Artificial
# 
# * **Prof. Cleilton Lima Rocha**
# * **email:** climarocha@gmail.com
# * **deadline: 13/04 às 23:59h**
# 
# 
# Para este trabalho vamos usar o [Data Set do Titanic disponível no Kaggle](https://www.kaggle.com/c/titanic).  
# 
# Vamos criar um modelo de classificação de sobreviventes e não sobreviventes utilizando a implementação da Árvore de Decisões e Random Fortest em Python para classificação. Mas antes disso vamos realizar algumas atividades, tais como EDA (Exploração de Dados) e o processo de pré-processamento dos dados.
# 
# 
# ## O Dicionário de Dados
# 
# **Survival**: Sobrevivente (Não=0,Sim=1)
# **Pclass**: Classe de ingresso (1=1st,2=2nd,3=3rd)
# **Sex**: Sexo
# **Age**: Idade em anos
# **Sibsp**: Quantidade de irmãos ou cônjuge a bordo do Titanic
# **Parch**: Quantidade de pais ou filhos a bordo do Titanic
# **Ticket**: Número do ticket
# **Fare**: Tarifa do passageiro
# **Cabin**: Número da cabine	
# **Embarked**: Portão de Embarque (C=Cherbourg, Q=Queenstown, S=Southampton)
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

# ## Import as bibliotecas
# Vamos importar algumas bibliotecas para começar!

# In[181]:


#Importando o dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ## Acessando os dados
# 
# Vamos iniciar lendo os dados através do pandas
# 
# - Utilizearemos a função **pd.read_csv** para ler o dado e salve na variável **data**
# - O nome do dataset é **titanic_train.csv**
# - Visualize alguns elementos do seu data set, para isto use **data.head()**

# In[182]:


df = pd.read_csv('C:/Users/Ernesto/trabalho_v1_classificacao/dataset/titanic_train.csv')


# In[183]:


df


# In[184]:


print(df.head())


# In[185]:


df['Ticket'][0:2]


# In[186]:


df['Survived'][0:2]


# ## Pré-processamento dos dados
# 

# #### Preenchimento do atributo idade

# P.S.: Queremos preencher os dados que faltam em algumas amostras para o atributo Idade. 
# Uma maneira de fazer isso é preencher com o valor da idade mediana de todos os passageiros ou realizarmos um filtro pela classe e considerarmos a mediana por classe. Selecione uma das opções e preencha o atributo.
# 
# P.S.: Dependo da escolha vc pode usar as funções fillna() ou apply()

# In[187]:


print(df.describe()) 
#Utilizando a função describe para obter dados do data set e ter uma noção dos numeros, valores, medias ete 
#Exemplo media da idade é 29 anos
#Se os dados estao dispersos ou não e etc


# In[188]:


#Confirmar atraves de uma visualização como os dados estão dispostos atraves de uma media de idade
# histograma
df.Age.hist(bins = 30)
plt.xlabel("Idade Pessoa")
plt.ylabel("Quantidade de Pessoas")
plt.title("Distribuição Relativa da Idade")
plt.show()


# In[189]:


#Verificação agora usando codifo para ter certeza
df.isnull().values.any()


# In[190]:


#Indicação True realmente confirma que existem valores missing


# In[191]:


#Verificando os dados nulos para serem tratados
#pd.isnull(df) verificando o data set inteiro
pd.isnull(df)


# In[192]:


#Olhando o conjunto de dados é possivel perceber que alguns dados além da idade precisam ser tratados 
#porem vamos nos concentrar somente no conjunto de dados de idade como a questao pedia


# In[193]:


pd.isnull(df["Age"])


# In[194]:


print(df.describe())
#Media da idade: Age; 29.699118 


# In[195]:


#Em uma analise exploratoria fazendo a contagem total de linhas do Data Frame e fazendo uma analise detalhada e exploratoria
#de cada tipo de dado faltante em quantidade sobreviventes e mortos
#Utilizando como ideia contagem de linhas contendo determinado valor
#total de 549 nao sobreviventes - 891 -> 342
print("# Quantidade de rows no dataframe {0}".format(len(df)))
print("# Survived 'Zero Não': {0}".format(len(df.loc[df['Survived'] == 0])))


# In[196]:


#tratando valores missing ------- inicio do tratamento
# importando preprocessador do pactoe para então dizer os valores faltantes para então serem substituidos
import sklearn as sk
from sklearn.preprocessing import Imputer


# In[197]:


# Criando objeto para substituição dos valores que faltam pela media, eles serao posterirmente inseridos dessa maneira
# nos dados de treino
preenche_Idade = Imputer(missing_values=np.nan,strategy='mean', axis = 0)


# Substituindo os valores iguais a zero, pela média dos dados (Sera mostrado mais a frente)
#X_treino = preenche_0.fit_transform(X_treino)
#X_teste = preenche_0.fit_transform(X_teste)


# #### Remova os seguintes atributos: Name, Ticket, PassengerId e Cabin
#  *Dica:* use a função drop, por exemplo, df.drop(['colunas'],axis=1,inplace=True)

# In[201]:


#removendo a coluna apenas comentada porque quando ja removida da erro
df.drop(['Name'],axis=1,inplace=True)


# In[202]:


#removendo a coluna apenas comentada porque quando ja removida da erro
df.drop(['Ticket'],axis=1,inplace=True)


# In[203]:


#removendo a coluna apenas comentada porque quando ja removida da erro
df.drop(['PassengerId'],axis=1,inplace=True)


# In[204]:


#removendo a coluna apenas comentada porque quando ja removida da erro
df.drop(['Cabin'],axis=1,inplace=True)


# In[205]:


#Nova Estrutura do DataSet (Data Frame tratado)
df.fillna(value=29.7)
#inserindo o valor 29.7 por ser a media arredonda do valor entregue da media anteriormente
# mean > 29.699


# #### Verifique se há valores nulos df.isnull() e remova os valores nulos() com a função df.dropna(inplace=True)

# In[206]:


#verificado se existem valores nulos
df.isnull()


# In[207]:


#verificando novamente por linha de comando
df.isnull().values.any()


# In[208]:


#usando comando para retirar outros quaisquer possiveis valores. Embora não existam, 'porem deixar comando salvo'
df.dropna(inplace=True)


# ### Exemplo

# Veja o exemplo do One Hot Enconding

# In[16]:


#sex = pd.get_dummies(data['Sex'], prefix='Gender', drop_first=True)
#data.drop(['Sex'],axis=1,inplace=True)
#data = pd.concat([data,sex],axis=1)


# #### Veja o exemplo acima e refaça o mesmo processo para o atributo *Embarked* e *Cabin*, visualize como os seus dados ficaram

# In[167]:


from sklearn.model_selection import train_test_split


# In[209]:


# Executando para os arquivos em data para:  <- Descomentar se necessario executar novamente
#SEXO
sex = pd.get_dummies(df['Sex'], prefix='Gender', drop_first=True)
df.drop(['Sex'],axis=1,inplace=True)
df = pd.concat([df,sex],axis=1)


# In[210]:


# Executando para os arquivos em data para:  <- Descomentar se necessario executar novamente
#Embarked
Embarked = pd.get_dummies(df['Embarked'], prefix='Condicao', drop_first=True)
df.drop(['Embarked'],axis=1,inplace=True)
df = pd.concat([df,sex],axis=1)


# In[211]:


# Executando para os arquivos em data para:  <- Descomentar se necessario executar novamente
#Cabin <- Cabin não ira executar porque foi pedido na questao anterior para ser removida ja
#Cabin = pd.get_dummies(df['Cabin'], prefix='Local', drop_first=True)
#df.drop(['Sex'],axis=1,inplace=True)
#df = pd.concat([df,sex],axis=1)


# ## Construindo os modelos de Árvore de Decisão

# Vamos começar dividindo nossos dados em um conjunto de treinamento e conjunto de testes. Você sabe o que são esses datasets?

# ### Divisão Test-Train

# In[242]:


#Iniciando a divisao do data Set
#X = data.drop('Survived',axis=1)
#Y = data['Survived']
X = df.drop('Survived',axis=1)
Y = df['Survived']


# In[243]:


#Dividindo os dados do modelo
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=101)


# In[245]:


#Dados de treino
X_train


# In[247]:


# Resposta para a pergunta: 
#Você sabe o que são esses datasets?
# Resposta: Sim, são os conjuntos de dados que eu vou usar para treinar o modelo e testar o modelo! No caso fazendo a separação
# deles!


# ### Treinandos os modelos

# #### Importe a classe [DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) e [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) do Sklearn. Crie um dos modelos para um dos algoritmos.

# In[275]:


#Importando as bliotecas
from sklearn.tree import DecisionTreeClassifier


# In[276]:


#iniciando o modelo
dtc = DecisionTreeClassifier()


# #### 10. Treine o modelo chamando a função fit(x_train,y_train)

# In[277]:


#treinando o modelo
dtc.fit(X_train, Y_train)


# #### 11. Faça as predições chamando a função predict(x_test)

# In[287]:


predictions = dtc.predict(X_test)


# In[289]:


X_test[0:5]


# In[290]:


X_train[0:5]


# In[291]:


Y_test[0:5]


# In[292]:


Y_train[0:5]


# In[280]:


#chamando os resultados das predições
predictions[0:]


# In[297]:


predictions_proba = dtc.predict_proba(X_test)


# In[298]:


#verificando as probabilidades
print(predictions_proba)


# ## Avaliação

# #### Matriz de confusão

# In[299]:


from sklearn.metrics import classification_report, confusion_matrix


# In[300]:


import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# In[303]:


cnf_matrix = confusion_matrix(Y_test, predictions)
cnf_matrix


# In[305]:


#print para mostrar o resultado de identificação quão bom foi o modelo e sua precisão
print(classification_report(Y_test,predictions))


# #### Plot a matriz de confusão usando a função acima

# In[307]:


#criação da label da matriz
cnf_matrix = confusion_matrix(Y_test, predictions, labels=[1,0])


# In[308]:


plot_confusion_matrix(cnf_matrix, classes=['Sobrevivente','Nao sobrevivente'],
                      title='Matriz de confusao')


# #### Podemos verificar a precisão, o recall, o f1-score usando o relatório de classiicação! 

# In[309]:


#print para mostrar o resultado de identificação quão bom foi o modelo e sua precisão
print(classification_report(Y_test,predictions))


# ### Desafios

# * ####  Qual o valor curva ROC AUC?

# In[397]:


#O ROC possui dois parâmetros:
#Taxa de verdadeiro positivo (True Positive Rate), que é dado por true positives / (true positives + false negatives)
#Resposta: Positivo/ Positivo  + Falso Positivo
#Taxa de falso positivo (False Positive Rate), que é dado por false positives / (false positives + true negatives)
#Resposta: Falso Positivo/ Positivo  + Falso Positivo


# In[400]:


# Import some data to play with
iris = datasets.load_iris()
x = df.drop('Survived',axis=1)
y = df['Survived']

# Binarize the output
y = label_binarize(y, classes=[0, 1, 2])
n_classes = y.shape[1]


# In[401]:


plt.figure()
lw = 2
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# * #### Plot as features mais importantes 

# In[356]:


#lista das features
print("Pclass → Age → SibSp → Parch → Fare → Gender_male → Gender_male")
df.head()


# In[354]:


#Features importantes em valor
dtc.feature_importances_


# In[359]:


#importando uma lib
from pandas import DataFrame


# In[362]:


data = {'Pclass': [0.11819133],
        'Age': [0.29196888],
        'SibSp': [0.04953841],
        'Parch:': [0.02271113],
        'Fare:': [0.21307227],
        'Gender_male:':[0.30451797],
        'Gender_male:': [0.        ]}


# In[364]:


frame = DataFrame(data)


# In[365]:


frame


# * #### Plot a árvore de decisão
