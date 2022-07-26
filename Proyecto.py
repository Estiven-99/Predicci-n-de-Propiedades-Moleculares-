#Primero importamos las librerias
####################################################
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import make_scorer,mean_squared_error
from sklearn.model_selection import cross_val_score
import random
random.seed(42)
import os
print(os.listdir("../input"))



#Necesitamos los siguientes datos
####################################################

pot_energy=pd.read_csv('../input/potential_energy.csv')
mulliken_charges=pd.read_csv('../input/mulliken_charges.csv')
train_df=pd.read_csv('../input/train.csv')
scalar_coupling_cont=pd.read_csv('../input/scalar_coupling_contributions.csv')
test_df=pd.read_csv('../input/test.csv')
magnetic_shield_tensor=pd.read_csv('../input/magnetic_shielding_tensors.csv')
dipole_moment=pd.read_csv('../input/dipole_moments.csv')
structures=pd.read_csv('../input/structures.csv')




print('Shape of potential energy dataset:',pot_energy.shape)
print('Shape of mulliken_charges dataset:',mulliken_charges.shape)
print('Shape of train dataset:',train_df.shape)
print('Shape of scalar coupling contributions dataset:',scalar_coupling_cont.shape)
print('Shape of test dataset:',test_df.shape)
print('Shape of magnetic shielding tensors dataset:',magnetic_shield_tensor.shape)
print('Shape of dipole moments dataset:',dipole_moment.shape)
print('Shape of structures dataset:',structures.shape)




#Exploramos los datasetes
####################################################


#Dataset de energia
print('Data Types:\n',pot_energy.dtypes)
print('Descriptive statistics:\n',np.round(pot_energy.describe(),3))
pot_energy.head(6)



#Datasetes de la carga
print('Data Types:\n',mulliken_charges.dtypes)
print('Descriptive statistics:\n',np.round(mulliken_charges.describe(),3))
mulliken_charges.head(6)


#Datasetes 
print('Data Types:\n',train_df.dtypes)
print('Descriptive statistics:\n',np.round(train_df.describe(),3))
train_df.head(6)



#Datasetes del acople escalar
print('Data Types:\n',scalar_coupling_cont.dtypes)
print('Descriptive statistics:\n',np.round(scalar_coupling_cont.describe(),3))
scalar_coupling_cont.head(6)

#Dataset estadistico
print('Data Types:\n',test_df.dtypes)
print('Descriptive statistics:\n',np.round(test_df.describe(),3))
test_df.head(6)

#Dataset de tensor de campo magnetico
print('Data Types:\n',magnetic_shield_tensor.dtypes)
print('Descriptive statistics:\n',np.round(magnetic_shield_tensor.describe(),3))
magnetic_shield_tensor.head(6)


#Dataset de la estructura
print('Data Types:\n',structures.dtypes)
print('Descriptive statistics:\n',np.round(structures.describe(),3))
structures.head(6)

#######################################################################################
#Mapa de la estructura atomica y prueba

def map_atom_data(df,atom_idx):
    df=pd.merge(df,structures,how='left',
               left_on=['molecule_name',f'atom_index_{atom_idx}'],
               right_on=['molecule_name','atom_index'])
    df=df.drop('atom_index',axis=1)
    df=df.rename(columns={'atom':f'atom_{atom_idx}',
                         'x':f'x_{atom_idx}',
                         'y':f'y_{atom_idx}',
                         'z':f'z_{atom_idx}'})
    return df

train_df['type_0']=train_df['type'].apply(lambda x:x)
test_df['type_0']=test_df['type'].apply(lambda x : x)

train_df=train_df.drop(columns=['molecule_name','type'],axis=1)
display(train_df.head(6))


test_df=test_df.drop(columns=['molecule_name','type'],axis=1)
display(test_df.head(10))


##############################################
#Histograma de visualizacion

train_df['type_0']=train_df.type_0.astype('category')
train_df['atom_0']=train_df.atom_0.astype('category')
train_df['atom_1']=train_df.atom_1.astype('category')


test_df['type_0']=test_df.type_0.astype('category')
test_df['atom_0']=test_df.atom_0.astype('category')
test_df['atom_1']=test_df.atom_1.astype('category')

plt.hist(train_df['scalar_coupling_constant'])
plt.ylabel('No of times')
plt.xlabel('scalar copling constant')
plt.show()

plt.hist(train_df['dist_vector'])
plt.ylabel('No of times')
plt.xlabel('Distance vector')
plt.show()

plt.hist(train_df['dist_X'])
plt.ylabel('No of times')
plt.xlabel('X distance vector')
plt.show()


plt.hist(train_df['dist_Y'])
plt.ylabel('No of times')
plt.xlabel('Y distance vector')
plt.show()


plt.hist(train_df['dist_Z'])
plt.ylabel('No of times')
plt.xlabel('Z distance vector')
plt.show()

train_df.head(5)

#########################################################333
#Modelo de prediccion 



threshold=0.95
corr_matrix=train_df.corr().abs()

upper=corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(np.bool))

to_drop=[column for column in upper.columns if any(upper[column]>threshold)]
print('There are are %d columns to remove.'%(len(to_drop)))

train_df=train_df.drop(columns=to_drop)
test_df=test_df.drop(columns=to_drop)
print('Training data shape',train_df.shape)
print('Testing data shape',test_df.shape)


Attributes=['atom_index_0','atom_index_1','type_0','x_0','y_0','z_0','atom_0',
            'atom_1','x_1','y_1','z_1','dist_vector','dist_X','dist_Y','dist_Z']

cat_attributes=['type_0','atom_0','atom_1']
target_label=['scalar_coupling_constant']


X_train=train_df[Attributes]
X_test=test_df[Attributes]
y_target=train_df[target_label]

X_train=pd.get_dummies(data=X_train,columns=cat_attributes)
X_test=pd.get_dummies(data=X_test,columns=cat_attributes)

print(X_train.shape,X_test.shape)
display(y_target.shape)



from sklearn.preprocessing import LabelEncoder
for f in ['type','atom_index_0','atom_index_1','atom_0','atom_1']:
    if f in good_columns:
        lbl=LabelEncoder()
        lbl.fit(list(X_train[f].values)+list(X_test[f].values))
        X_train[f]=lbl.transform(list(X_train[f].values))
        X_test[f]=lbl.transform(list(X_test[f].values))



X_train.head(6)
X_test.head(6)
y_target.head(6)


linear_reg=linear_model.LinearRegression()
n_folds=5
lin_reg_score=cross_val_score(linear_reg,X_train,y_target,
                          scoring=make_scorer(mean_squared_error),
                          cv=n_folds)
lin_score=sum(lin_reg_score)/n_folds
print('Lin_score:',lin_score)  


lr_model=linear_reg.fit(X_train,y_target)
score=np.round(lr_model.score(X_train,y_target),3)
print('Accuracy of trained model:',score)
model_coeff=np.round(lr_model.coef_,3)
print('Model coefficients:',model_coeff)
model_intercept=np.round(lr_model.intercept_,3)
print('Model intercept value:',model_intercept)

from sklearn.metrics import r2_score
y_pred=lr_model.predict(X_test)
SCC=pd.read_csv('../input/sample_submission.csv')
SCC['scalar_coupling_constant']= y_pred
SCC.to_csv('Linear_Regression_model.csv',index=False)



#continua