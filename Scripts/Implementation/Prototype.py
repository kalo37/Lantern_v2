'''                     No School Violence
                        Distribution 1.0
                       24 September 2017
                           Harsh Shukla     
                  harsh.shukla@u.northwestern.edu

I. Introduction
   This is a working prototype of an NLP model which identifies related articles and uses iterative process to
   come up with the best model parameters before model building 

II. Model

   This code starts with stemming of the text and uses n-gram approach to add the features in the model. Next, feature consolidation is carried out 
   by using Truncated SVD which is a linear dimensionality reduction technique for sparse matrices.
   
   Here are the steps followed by this algorithm
   
   1) Read the data, perform stemming (snowball stemmer is used for stemming) and divide it into test and training 
   2) Perform vectorization
   3) Perform Truncated SVD for feature reduction 
   4) Compute TF-IDF values
   5) Perform supervised Nearest neighbur classifier algorithm
   6) Use the best parameters found by the iterative process to build model on the test data set

   Currently the project is using tf-idf process to find the important keywords within an article
   which can be used for either document classification or document retrial. A key challenge with
   model process is of choosing the right parameters. For example, in TF-idf process we can set
   a threshold for Document frequencies or use l1 or l2 norm of normalization and many more paramenters 
   which can substantially influence the model performance. This code performs an iterative search for every step (as mentioned above) 
   in deciding the best model parameters and uses them in model building.


   *** Note - If you are running this code on your local machine, make sure you do not put in to many parameters in the pipeline step.
   It may cause your system to hang because it is a computationally itensive process. Ideally, this code should run on multicore cluster.

   ****Kindly add comments here when adding to this code and revise the Distribution version


'''

from __future__ import print_function
from pprint import pprint
from time import time
import logging
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
stem = SnowballStemmer('english')
categories = None
data = fetch_20newsgroups(subset='train', categories=categories)
#data_stem = data.data.map(lambda x: ' '.join([stemmer.stem(y) for y in x.split(' ')]))
for indx,i in enumerate(data.data):
    data.data[indx] = (' '.join([stem.stem(y) for y in i.split(' ')]))
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.30, random_state=50)
print("%d documents" % len(data.filenames))
print("%d categories" % len(data.target_names))
print()

# #############################################################################
pipeline = Pipeline([
    ('vect', CountVectorizer(encoding='utf-8',stop_words='english')),
    ('pca', TruncatedSVD()),
    ('tfidf', TfidfTransformer()),
    ('clf', KNeighborsClassifier()),
])

# uncommenting more parameters will give better exploring power but will
# increase processing time in a combinatorial way
parameters = {
    #'vect__max_df': (0.5, 0.75, 1.0),
    #'vect__max_features': (None, 5000, 10000, 50000),
    #'vect__ngram_range': ((1,1),(1,1)),  # unigrams or bigrams
    'pca__n_components':(500,1000),
    #'pca__n_iter':(5),
    #'pca__n_components':('arpack','randomized')
    #'tfidf__use_idf': (True, False),
    #'tfidf__norm': ('l1', 'l2'),
    ##'clf__alpha': (0.00001, 0.000001),
    ##'clf__penalty': ('l2', 'elasticnet'),
    #'clf__algorithm': ('ball_tree','kd_Tree','brute,auto'),
    'clf__n_neighbors':(3,4)
    
    #'clf__n_iter': (10, 50, 80),
}

if __name__ == "__main__":

    # find the best parameters for both the feature extraction and the unsupervised KNN search
    
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1 , cv =5)
    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    grid_search.fit(X_train, y_train)
    print("done in %0.3fs" % (time() - t0))
    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    model_params = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, model_params[param_name]))
    
    ###################################  Running the algorithm on the test data set ####################
    print("\n Running unsupervised Knn on test dataset \n ")
    tfidf_mod = TfidfVectorizer(stop_words='english', ngram_range=(1,1), norm = model_params['tfidf__norm'], use_idf = model_params['tfidf__use_idf']
    , smooth_idf = model_params['tfidf__smooth_idf'],sublinear_tf = model_params['tfidf__sublinear_tf'])
    tfidf_weights = tfidf_mod.fit_transform(X_test)
    pca_test = TruncatedSVD(n_components = model_params['clf__n_neighbors'])
    pca_test.fit(tfidf_weights)
    pca_trans = pca_test.transform(tfidf_weights)
    neigh = KNeighborsClassifier(n_neighbors=model_params['clf__n_neighbors'],p=model_params['clf__p'])
    neigh.fit(pca_trans,y_test)
    print ('The mean accuracy (in percentage) on the test set is given as %0.3f \n'% (100 * neigh.score(pca_trans,y_test)))
    for count,i in enumerate(neigh.kneighbors(pca_test.transform(tfidf_weights[201]),3,False).tolist()[0]):
        print('The matching text # %d is: \t %s' % (count , X_test[i]))