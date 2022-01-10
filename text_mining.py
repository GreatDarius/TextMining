#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# Import libraries
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# Import Dataset
data = pd.read_csv("tweet_DM.csv")

# drop blanck rows
data.dropna(inplace=True)

# Shuffeling the data
data = data.sample(frac = 1)

# Change the gender column to binary
data.loc[data['gender']=='Male', 'gender']=0
data.loc[data['gender']=='Female', 'gender']=1
# Make it integer
data['gender']=data['gender'].astype('int')
# Select the target and input variables
target = data['gender']

# Check for missing values
data['text'].isna().sum()
data['text'].fillna('missing', inplace=True)
input_data = data['text']

# Text preprocessing
nltk.download('stopwords')
nltk.download('wordnet')

# For each row in input_data, we will read the text, tokenize it, remove stopwords, lemmatize it
new_text = []
for text in input_data:
	text = re.sub(r'[!"#$%&()*+,-./:;<=>?[\]^_`{|}~]@', ' ', text).lower()

	words= nltk.tokenize.word_tokenize(text)
	words = [w for w in words if w.isalpha()]
	words = [w for w in words if len(w)>2 and w not in stopwords.words('english')]

	lemmatizer = nltk.stem.WordNetLemmatizer()
	words = [lemmatizer.lemmatize(w) for w in words]
	new_text.append(' '.join(words))

# Split the data into train and test set
X_train, X_test, y_train, y_test = train_test_split(new_text, target, test_size=0.25, random_state=42, stratify=target)

# Countvectorizer includes pre-processing, tokenization, filtering stop words
# Let's see if we can limite the features and achieve a good accuracy
count_vect = CountVectorizer()

train_x_tr = count_vect.fit_transform(X_train)
test_x_tr = count_vect.transform(X_test)

# TFIDF
tf_transformer = TfidfTransformer()
train_x_tfidf = tf_transformer.fit_transform(train_x_tr)
#train_x_tfidf.shape
test_x_tfidf = tf_transformer.transform(test_x_tr)
#test_x_tfidf.shape
#df_tfidf = pd.DataFrame(train_x_tfidf.toarray())

# If you are performing Latent Semantic Analysis, recommended number of components is 100
svd = TruncatedSVD(n_components=676)
train_x_lsa = svd.fit_transform(train_x_tfidf)
svd.explained_variance_.sum()

# Let's select the first component
first_component = svd.components_[0,:]
# Sort the weights in the first component, and get the indeces
import numpy as np
indeces = np.argsort(first_component).tolist()
print(indeces)

feat_names = count_vect.get_feature_names()
for index in indeces[-10:]:
    print(feat_names[index], "\t\tweight =", first_component[index])

test_x_lsa = svd.transform(test_x_tfidf)

# Calculate the baseline
# Find majority class
y_train.value_counts()
# Find percentage
y_train.value_counts()/len(y_train)

# Model 1: Naive Bayes
mnb = MultinomialNB()
mnb.fit(train_x_tfidf, y_train)

# Predict the train values
train_y_pred_mnb = mnb.predict(train_x_tfidf)
# Train accuracy
mnb_acc_train = accuracy_score(y_train, train_y_pred_mnb)
# Predict the test values
test_y_pred_mnb = mnb.predict(test_x_tfidf)
# Test accuracy
mnb_acc_test = accuracy_score(y_test, test_y_pred_mnb)
#We usually create the confusion matrix on test set
mnb_cm = confusion_matrix(y_test, test_y_pred_mnb)

# Tunning
from sklearn.model_selection import GridSearchCV
import numpy as np
grid_params = {
  'alpha': np.linspace(0.0001, 1.5, 1500),
  'fit_prior': [True, False],
}
clf = GridSearchCV(mnb, grid_params)
clf.fit(train_x_tfidf, y_train)
print("Best Score: ", clf.best_score_)
print("Best Params: ", clf.best_params_)

# Model 2: Random Forest
from sklearn.ensemble import RandomForestClassifier
rndf = RandomForestClassifier(n_estimators=266, min_samples_split=2, min_samples_leaf=1,
                              max_features='sqrt', bootstrap=False, n_jobs=-1)
rndf.fit(train_x_lsa, y_train)
# Train accuracy
train_y_pred_rndf = rndf.predict(train_x_lsa)
rndf_acc_train = accuracy_score(y_train, train_y_pred_rndf)
# Test accuracy
test_y_pred_rndf = rndf.predict(test_x_lsa)
rndf_acc_test = accuracy_score(y_test, test_y_pred_rndf)
# Confusion matrix
confusion_matrix(y_test, test_y_pred_rndf)

# Tunning parameters
from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 50, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)
# Use the random grid to search for best hyperparameters
# First create the base model to tune
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rndf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42)
# Fit the random search model
rf_random.fit(train_x_lsa, y_train)
def evaluate(model, test_features, test_labels):
    # Test accuracy
    test_y_pred_rndf = rndf.predict(test_features)
    rndf_acc_test = accuracy_score(test_labels, test_y_pred_rndf)
    # Confusion matrix
    confusion_matrix(test_labels, test_y_pred_rndf)

    print('Model Performance')
    print('Accuracy = {:2f}.'.format(rndf_acc_test))

    return rndf_acc_test
base_model = RandomForestClassifier(random_state = 42)
base_model.fit(train_x_lsa, y_train)
base_accuracy = evaluate(base_model, test_x_lsa, y_test)
best_random = rf_random.best_estimator_
random_accuracy = evaluate(best_random, test_x_lsa, y_test)
print('Improvement of {:2f}.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))


# Model 3: Stochastic Gradient Descent Classifier
from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier()
sgd.fit(train_x_lsa, y_train)
# Train accuracy
train_y_pred_sgd = sgd.predict(train_x_lsa)
sgd_acc_train = accuracy_score(y_train, train_y_pred_sgd)
# Test accuracy
test_y_pred_sgd = sgd.predict(test_x_lsa)
sgd_acc_test = accuracy_score(y_test, test_y_pred_sgd)
# Confusion matrix
confusion_matrix(y_test, test_y_pred_sgd)


# Model 4: Descision Tree
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=17, min_samples_leaf=7)
tree.fit(train_x_lsa, y_train)

train_y_pred_tree = tree.predict(train_x_lsa)
tree_acc_train = accuracy_score(y_train, train_y_pred_tree)
# Test accuracy
test_y_pred_tree = tree.predict(test_x_lsa)
tree_acc_test = accuracy_score(y_test, test_y_pred_tree)
# Confusion matrix
confusion_matrix(y_test, test_y_pred_tree)

# Tunning
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
param_grid = {'max_depth': randint(low=5, high=20),
              'min_samples_leaf': randint(low=5, high=20)}

tree_gs = RandomizedSearchCV(DecisionTreeClassifier(), param_grid,
                             n_iter=15, cv=5, verbose=1,
                             scoring='accuracy',
                             return_train_score=True)
tree_gs.fit(train_x_lsa, y_train)
cvres = tree_gs.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(mean_score, params)
#Find the best parameter set
tree_gs.best_params_
tree_gs.best_estimator_

train_y_pred_treegs = tree_gs.predict(train_x_lsa)
treegs_acc_train = accuracy_score(y_train, train_y_pred_treegs)
# Test accuracy
test_y_pred_treegs = tree_gs.predict(test_x_lsa)
treegs_acc_test = accuracy_score(y_test, test_y_pred_treegs)
# Confusion matrix
confusion_matrix(y_test, test_y_pred_treegs)

# Model 5: Neural Network
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(600,400,200),
                       max_iter=1000, tol = 0.00001)
mlp.fit(train_x_tfidf, y_train)
# Train accuracy
train_y_pred_mlp = mlp.predict(train_x_tfidf)
mlp_acc_train = accuracy_score(y_train, train_y_pred_mlp)
# Test accuracy
test_y_pred_mlp = mlp.predict(test_x_tfidf)
mlp_acc_test = accuracy_score(y_test, test_y_pred_mlp)
# Confusion matrix
confusion_matrix(y_test, test_y_pred_mlp)

from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import classification_report
file = open("result.txt","w+")
file.write("Model 1: Naive Bayes \n \n")
precision, recall, fscore, support = score(y_test, test_y_pred_mnb)
file.write('{}'.format(classification_report(y_test, test_y_pred_mnb, target_names=['Male', 'Female'])))
file.write('\nconfusion matrix: \n')
file.write('{}'.format(confusion_matrix(y_test, test_y_pred_mnb)))


file.write("\n \nModel 2: Random Forest \n \n")
precision, recall, fscore, support = score(y_test, test_y_pred_rndf)
file.write('{}'.format(classification_report(y_test, test_y_pred_rndf, target_names=['Male', 'Female'])))
file.write('\nconfusion matrix: \n')
file.write('{}'.format(confusion_matrix(y_test, test_y_pred_rndf)))

file.write("\n \nModel 3: Stochastic Gradient Descent Classifier \n \n")
precision, recall, fscore, support = score(y_test, test_y_pred_sgd)
file.write('{}'.format(classification_report(y_test, test_y_pred_sgd, target_names=['Male', 'Female'])))
file.write('\nconfusion matrix: \n')
file.write('{}'.format(confusion_matrix(y_test, test_y_pred_sgd)))

file.write("\n \nModel 4: Decision Tree \n \n")
precision, recall, fscore, support = score(y_test, test_y_pred_tree)
file.write('{}'.format(classification_report(y_test, test_y_pred_tree, target_names=['Male', 'Female'])))
file.write('\nconfusion matrix: \n')
file.write('{}'.format(confusion_matrix(y_test, test_y_pred_tree)))

file.write("\n \nModel 5: Neural Network \n \n")
file.write('{}'.format(classification_report(y_test, test_y_pred_mlp, target_names=['Male', 'Female'])))
file.write('\nconfusion matrix: \n')
file.write('{}'.format(confusion_matrix(y_test, test_y_pred_mlp)))
file.close()
