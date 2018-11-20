import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

df = pd.read_csv("C:\\Users\\User\\Downloads\\LoanPrediction.csv")

df['Gender'].replace(['Female','Male'],[0,1],inplace=True)
df['Married'].replace(['No','Yes'],[0,1],inplace=True)
df['Education'].replace(['Graduate','Not Graduate'],[0,1],inplace=True)
df['Self_Employed'].replace(['No','Yes'],[0,1],inplace=True)
df['Loan_Status'].replace(['N','Y'],[0,1],inplace=True)
df['Property_Area'].replace(['Rural','Semiurban','Urban'],[0,1,2],inplace=True)
df['Dependents'].replace(['0','1','2','3+'],[0,1,2,3],inplace=True)
df['Loan_Amount_Term'].replace([12,36,60,84,120,180,240,300,360,480],[0,1,2,3,4,5,6,7,8,9],inplace=True)
df=df.drop("Loan_ID",axis=1)

#df.info()
#print(df.describe())

df['IncomeBand'] = pd.cut(df['ApplicantIncome'], 7)

df['ApplicantIncome'] = np.where(df['ApplicantIncome'] <= 11700, 0, df['ApplicantIncome'])
df['ApplicantIncome'] = np.where(df['ApplicantIncome'].between(11701,23250), 1, df['ApplicantIncome'])
df['ApplicantIncome'] = np.where(df['ApplicantIncome'].between(23251,34800), 2, df['ApplicantIncome'])
df['ApplicantIncome'] = np.where(df['ApplicantIncome'].between(34801,46350), 3, df['ApplicantIncome'])
df['ApplicantIncome'] = np.where(df['ApplicantIncome'].between(463501,57900), 4, df['ApplicantIncome'])
df['ApplicantIncome'] = np.where(df['ApplicantIncome'].between(57901,69450), 5, df['ApplicantIncome'])
df['ApplicantIncome'] = np.where(df['ApplicantIncome'] > 69450, 6, df['ApplicantIncome'])
df['ApplicantIncome'] = np.where(df['ApplicantIncome'] == 51763, 4, df['ApplicantIncome'])

df=df.drop("IncomeBand",axis=1)

df['IncomeBand'] = pd.cut(df['CoapplicantIncome'], 4)

df['CoapplicantIncome'] = np.where(df['CoapplicantIncome'] <= 10416.75, 0, df['CoapplicantIncome'])
df['CoapplicantIncome'] = np.where(df['CoapplicantIncome'].between(10416.75001,20833.5), 1, df['CoapplicantIncome'])
df['CoapplicantIncome'] = np.where(df['CoapplicantIncome'].between(20833.5001,31250.25), 2, df['CoapplicantIncome'])
df['CoapplicantIncome'] = np.where(df['CoapplicantIncome'] > 31250.25, 3, df['CoapplicantIncome'])

df=df.drop("IncomeBand",axis=1)

df['IncomeBand'] = pd.cut(df['LoanAmount'], 6)

df['LoanAmount'] = np.where(df['LoanAmount'] <= 124.167, 0, df['LoanAmount'])
df['LoanAmount'] = np.where(df['LoanAmount'].between(124.16701,239.333), 1, df['LoanAmount'])
df['LoanAmount'] = np.where(df['LoanAmount'].between(239.33301,354.5), 2, df['LoanAmount'])
df['LoanAmount'] = np.where(df['LoanAmount'].between(354.501,469.667), 3, df['LoanAmount'])
df['LoanAmount'] = np.where(df['LoanAmount'].between(469.66701,584.833), 4, df['LoanAmount'])
df['LoanAmount'] = np.where(df['LoanAmount']>584.833,5, df['LoanAmount'])

df=df.drop("IncomeBand",axis=1)

#print(df[["Loan_Amount_Term", "Loan_Status"]].groupby(['Loan_Amount_Term'], as_index=False).mean().sort_values(by='Loan_Status', ascending=False))

print(df.corr())

#guess_gender = np.zeros((2))
#for dataset in df:
  #  for i in range(0,2):
        
   #                 guess_df_gender = df[
    #                               (df['Education']==i)]['Gender'].dropna()

                    # age_mean = guess_df.mean()
                    # age_std = guess_df.std()
                    # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

     #               gender_guess = guess_df_gender.median()

                    # Convert random age float to nearest .5 age
      #              guess_gender[i] = int( gender_guess/0.5 + 0.5 ) * 0.5
            
    #for i in range(0, 2):
     #       dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\
      #              'Age'] = guess_gender[i,j]

df=df.fillna(df.mean())
print(df.describe())

y=df["Loan_Status"]
x=df.drop("Loan_Status",axis=1)

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.33)

svc = SVC()
svc.fit(xtrain, ytrain)
ypred = svc.predict(xtest)

accuracy=accuracy_score(ytest,ypred)
print("Accuracy svm:",accuracy)

logreg = LogisticRegression()
logreg.fit(xtrain, ytrain)
ypred = logreg.predict(xtest)

accuracy=accuracy_score(ytest,ypred)
print("Accuracy logregression:",accuracy)

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(xtrain, ytrain)
Y_pred = knn.predict(xtest)

accuracy=accuracy_score(ytest,ypred)
print("Accuracy knn:",accuracy)

gaussian = GaussianNB()
gaussian.fit(xtrain, ytrain)
ypred = gaussian.predict(xtest)

accuracy=accuracy_score(ytest,ypred)
print("Accuracy gnb:",accuracy)

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(xtrain, ytrain)
ypred = random_forest.predict(xtest)

accuracy=accuracy_score(ytest,ypred)
print("Accuracy rf:",accuracy)

decision_tree = DecisionTreeClassifier()
decision_tree.fit(xtrain, ytrain)
ypred = decision_tree.predict(xtest)

accuracy=accuracy_score(ytest,ypred)
print("Accuracy dt:",accuracy)

linear_svc = LinearSVC()
linear_svc.fit(xtrain, ytrain)
ypred = linear_svc.predict(xtest)

accuracy=accuracy_score(ytest,ypred)
print("Accuracy lsvc:",accuracy)

perceptron = Perceptron()
perceptron.fit(xtrain, ytrain)
ypred = perceptron.predict(xtest)

accuracy=accuracy_score(ytest,ypred)
print("Accuracy perceptron:",accuracy)

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(12,12,12), random_state=1)
clf.fit(xtrain, ytrain)
ypred = clf.predict(xtest)

accuracy=accuracy_score(ytest,ypred)
print("Accuracy mlp:",accuracy)



