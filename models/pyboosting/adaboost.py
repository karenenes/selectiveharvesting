import pandas as pd
import matplotlib.pyplot as pl
import numpy as np
from six import string_types
from collections import defaultdict
from sklearn.metrics import roc_curve, auc, f1_score
from sklearn.tree import DecisionTreeClassifier

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def float_or_str(x):
    if isfloat(x):
        return (x)
    else:
        return (-1)


def percent_to_float(x):
    if isfloat(x):
        return (x/100)
    else:
        return float(x.strip('%'))/100

def add_noise(x):
    if not isinstance(x, string_types):
        # no noise needed for decision trees
        return (x)
    else:
        return (x)


def data_processing(filename):
    # Read file (must be in UTF-8 if using python version >= 3)
    df = pd.read_csv(filename)

    # print (df.head()) # check feature ids

    df['Interest Rate Percentage'] = [percent_to_float(i) for i in df['Interest Rate Percentage']]

    df['Debt-To-Income Ratio'] = [percent_to_float(i) for i in df['Debt-To-Income Ratio Percentage']]

    features_to_keep = ['Amount Requested','Interest Rate Percentage','Loan Purpose','Loan Length in Months',
                        'Monthly PAYMENT','Total Amount Funded','FICO Range','Debt-To-Income Ratio Percentage']

    # convert interger values to float (helps avoiding optimization implementation issues)
    for feature in features_to_keep:
        if feature not in ['FICO Range','Loan Purpose']:
            df[feature] = [float(i) for i in df[feature]]

    # Scale values
    df['Total Amount Funded'] /= max(df['Total Amount Funded'])
    df['Amount Requested'] /= max(df['Amount Requested'])
    df['Loan Length in Months'] /= max(df['Loan Length in Months'])
    df['Monthly PAYMENT'] /= max(df['Monthly PAYMENT'])

    # Interaction terms
    df['Total Amount Funded * Requested'] = df['Total Amount Funded']*df['Total Amount Funded']
    df['Total Amount Funded * Requested'] /= max(df['Total Amount Funded'])

    df['Interest Rate Percentage * Monthly PAYMENT'] = df['Interest Rate Percentage']*df['Monthly PAYMENT']
    df['Interest Rate Percentage * Monthly PAYMENT'] /= max(df['Interest Rate Percentage * Monthly PAYMENT'])


    target_var = [float_or_str(i) for i in df['Status']]

    # create a clean data frame for the regression
    data = df[features_to_keep].copy()

    data['intercept'] = 1.0

    return (data,target_var)

def add_categorical(train, test, feature_str):
    # encode categorical features
    encoded = pd.get_dummies(pd.concat([train[feature_str],test[feature_str]], axis=0))#, dummy_na=True)
    train_rows = train.shape[0]
    train_encoded = encoded.iloc[:train_rows, :]
    validation_encoded = encoded.iloc[train_rows:, :]

    train_encoded_wnoise = train_encoded.applymap(add_noise)
    validation_encoded_wnoise = validation_encoded.applymap(add_noise)

    train.drop(feature_str,axis=1, inplace=True)
    test.drop(feature_str,axis=1, inplace=True)

    train = train.join(train_encoded_wnoise.ix[:, :])
    test = test.join(validation_encoded_wnoise.ix[:, :])

    return (train,test)

class AdaBoost:
    def __init__(self,nWL,tree_depth=1):
        self.wls = []
        self.weights = []
        self.nWL = nWL
        self.tree_depth = tree_depth
        self.nfeats = 0

    def fit(self,X,y):
        # process input (lists)
        y = np.array(y,dtype=np.int16)
        y = y*2-1
        #print(y)

        X = np.array(X,dtype=np.float32)
        X = np.reshape(X,newshape=(len(y),-1),order='F')
        self.nfeats = (int)(X.size/y.size)
        #print(X)

		# input: dataset X and labels y (in {+1, -1})
        N = X.shape[0]
        w = np.ones(N) / N
        self.wls = []
        self.weights = []

        for m in range(self.nWL):
            w_learn = DecisionTreeClassifier(max_depth=self.tree_depth,criterion="entropy")

            norm_w = np.exp(w)
            norm_w /= norm_w.sum()
            w_learn.fit(X, y, sample_weight=norm_w)
            yhat = w_learn.predict(X)

            eps = norm_w.dot(yhat != y) + 1e-20
            alpha = (np.log(1 - eps) - np.log(eps)) / 2

            w = w - alpha * y * yhat

            self.wls.append(w_learn)
            self.weights.append(alpha)

    def print_all(self):
        print(self.wls)
        print(self.weights)

    def print_wls(self):
        print(self.wls)

    def print_weights(self):
        print(self.weights)

    def predict(self,X):
        # processing input (list)
        X = np.array(X,dtype=np.float32)
        X = np.reshape(X,newshape=(-1,self.nfeats),order='F')

        y = np.zeros(X.shape[0])
        for (w_learn, alpha) in zip(self.wls, self.weights):
            if not np.isnan(alpha):
                y = y + alpha * w_learn.predict(X)
        #return (np.sign(y))
        return y.tolist()

    def predict_and_print(self,X):
        # processing input (list)
        X = np.array(X,dtype=np.float32)
        X = np.reshape(X,newshape=(-1,self.nfeats),order='F')

        y = np.zeros(X.shape[0])
        for (w_learn, alpha) in zip(self.wls, self.weights):
            z = w_learn.predict(X)
            print(z)
            y = y + alpha * z
        #return (np.sign(y))
        return y.tolist()

    def fit_and_predict(self,X,y,newX):
        fit(X,y)
        return predict(newX)

# read the data in

#train_file = "Bank_Data_Train.csv"
#validation_file = "Kaggle_Public_Validation.csv"#"Bank_Data_Train.csv"
#
#train_sizes = [300, 500,1000,2000, 2500,3000, 3400]
#tree_depths = [1,10]
#nWL_range = [1,10,5000]
#
#
#f1_scores = defaultdict(lambda:defaultdict(lambda:[]))
#
#
#for train_size in train_sizes:
#    data_train, target_train = data_processing(train_file)
#    data_train = data_train[0:train_size]
#    target_train = target_train[0:train_size]
#
#    data_validation, target_validation = data_processing(validation_file)
#
#    # replace categorical strings with 1-of-K coding and add a small amount of Gaussian noise so it follows Gaussian model assumption
#
#    data_train, data_validation = add_categorical(train=data_train,test=data_validation,feature_str='FICO Range')
#
#    data_train, data_validation = add_categorical(train=data_train,test=data_validation,feature_str='Loan Purpose')
#
#
#    for nWL in nWL_range:
#        for tree_depth in tree_depths:
#            # Describe classifier and regularization type
#            X_train = data_train.as_matrix()
#            y_train = 2*np.array(target_train) - 1
#
#            X_validation = data_validation.as_matrix()
#            y_validation = 2*np.array(target_validation) - 1
#
#            adab = AdaBoost(nWL = nWL, tree_depth=tree_depth)
#            adab.fit(X=X_train, y=y_train)
#            y_pred = adab.predict(X=X_validation)
#
#            ### F1 Score ###
#            f1score = f1_score(target_validation, y_pred)
#            f1_scores[tree_depth][nWL].append(f1score)
#
#            print("New Tree Depth = %d" % tree_depth)
#            print("- F1 score: %f" % f1score)
#
#
#for wl in nWL_range:
#    for tree_depth in tree_depths:
#        pl.plot(train_sizes, f1_scores[tree_depth][wl], label='{wl} wl param; tree depth {trs}'.format(wl=wl,trs=tree_depth),ls='-',marker='o')
#
#
#pl.legend(loc='best')
#pl.xlabel('Train Samples')
#pl.ylabel('F1 Score')
#pl.ylim([0.4,0.8])
#pl.title('F1 Score x Tree Depth')
##ax.set_xticklabels([])
#
#pl.show()
