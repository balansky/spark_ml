from pyspark.ml.classification import RandomForestClassifier,NaiveBayes,DecisionTreeClassifier, \
    LogisticRegression,MultilayerPerceptronClassifier,OneVsRest
from pyspark.ml.classification import RandomForestClassificationModel,NaiveBayesModel,DecisionTreeClassificationModel,\
    LogisticRegressionModel,MultilayerPerceptronClassificationModel,OneVsRestModel


class Classifier():

    def __init__(self, cname):
        self.cname = cname

    def get(self, **kwargs):
        if self.cname == 'lr':
            clf = LogisticRegression(**kwargs)
        elif self.cname == 'rf':
            clf = RandomForestClassifier(**kwargs)
        elif self.cname == 'dt':
            clf = DecisionTreeClassifier(**kwargs)
        elif self.cname == 'nb':
            clf = NaiveBayes(**kwargs)
        elif self.cname == 'mp':
            clf = MultilayerPerceptronClassifier(**kwargs)
        else:
            lr = LogisticRegression(**kwargs)
            clf = OneVsRest(classifier=lr, featuresCol=kwargs['featuresCol'],
                            labelCol=kwargs['labelCol'], predictionCol=kwargs['predictionCol'])
        return clf

    def get_model(self,savePath):
        if self.cname == 'lr':
            clf = LogisticRegressionModel.load(savePath)
        elif self.cname == 'rf':
            clf = RandomForestClassificationModel.load(savePath)
        elif self.cname == 'dt':
            clf = DecisionTreeClassificationModel.load(savePath)
        elif self.cname == 'nb':
            clf = NaiveBayesModel.load(savePath)
        elif self.cname == 'mp':
            clf = MultilayerPerceptronClassificationModel.load(savePath)
        else:
            clf = OneVsRestModel.load(savePath)
        return clf



