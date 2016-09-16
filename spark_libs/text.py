from bs4 import BeautifulSoup
from pyspark.ml.feature import HashingTF,IDF,StringIndexer,StopWordsRemover,Word2Vec,IndexToString\
    ,StringIndexerModel,Word2VecModel
from pyspark.ml import Pipeline, PipelineModel
import re


class Tokenizer():

    def __init__(self,textFields,strategy):
        self.strategy = strategy
        self.textFields = textFields

    def combine_text_fields(self,row):
        combined_text = ''
        for text_field in self.textFields:
            combined_text += row[text_field] + ' '
        return combined_text

    def clean_html_tags(self,combined_text):
        soup = BeautifulSoup(combined_text, 'html.parser')
        clean_text = soup.get_text()
        return clean_text

    def pick_en_words(self,clean_text):
        words = []
        wordpattern = re.compile(r"[A-Za-z]{2,}([-_A-Za-z0-9]{2,})?")
        if wordpattern.search(clean_text):
            words = [w.group().lower() for w in wordpattern.finditer(clean_text)]
        return words

    def clean_combine_text(self,row):
        combined_text = self.combine_text_fields(row)
        clean_text = self.clean_html_tags(combined_text)
        return clean_text

    def tokenize_text(self,row):
        clean_text = self.clean_combine_text(row)
        words = self.pick_en_words(clean_text)
        return words

    def process(self, spark,df, outputCol,idCol):
        if self.strategy == 'remove':
            remover = StopWordsRemover(inputCol="dwords", outputCol=outputCol)
            cleantrainDF = remover.transform(spark.createDataFrame(df.rdd.map(
                lambda r: (r[idCol], self.tokenize_text(r))
            ), [idCol, "dwords"]))
        else:
            cleantrainDF = spark.createDataFrame(df.rdd.map(
                lambda r: (r[idCol], self.clean_combine_text(r))),[idCol,outputCol])
        cleantrainDF = cleantrainDF.join(df, idCol, 'inner')
        return cleantrainDF

class Vectorizer():

    def __init__(self, vname):
        self.vname = vname


    def tfidf(self, inputCol, outputCol, numFeatures, **kwargs):
        hashingTF = HashingTF(numFeatures=numFeatures, inputCol=inputCol, outputCol="hashvec")
        idf = IDF(inputCol="hashvec", outputCol=outputCol, **kwargs)
        pipline = Pipeline(stages=[hashingTF, idf])
        return pipline

    def get(self,**kwargs):
        if self.vname == "tfidf":
            model = self.tfidf(**kwargs)
        else:
            model = Word2Vec(**kwargs)
        # vecModel = model.fit(df)
        return model

    def get_model(self, modelPath):
        if self.vname == "tfidf":
            model = PipelineModel.load(modelPath)
        else:
            model = Word2VecModel.load(modelPath)
        return model



class Indexizer():

    def get(self, **kwargs):
        strindex = StringIndexer(**kwargs)
        return strindex

    def get_model(self, modelPath,outputCol, inputCol='prediction'):
        stringModel = StringIndexerModel.load(modelPath)
        inverterModel = IndexToString(inputCol=inputCol, outputCol=outputCol, labels=stringModel.labels)
        return inverterModel

    def get_inverter(self, model, outputCol, inputCol='prediction'):
        inverterModel = IndexToString(inputCol=inputCol, outputCol=outputCol, labels=model.labels)
        return inverterModel



