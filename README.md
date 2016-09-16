# spark_ml

The purpose Of this project is to demonstrate the usage of spark mllib from training to predicting ecommercial product's category.

Training Stage:
1.  Uses tfidf to vectorize text context
2.  Uses Cliffisier Algorithms such as Naive Bayes, Ovr Logistic Regression, Random Forest to train the model

Predict Stage:
1.  Load models from hdfs
2.  Use models to classify products' category
