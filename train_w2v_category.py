from pyspark.sql import SparkSession
from spark_libs import text, classifier, render
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

if __name__ == "__main__":
    sql_info = {
        "unique_id": "goods_uuid",
        "text_fields": ["goods_title", "goods_description"],
        "label_field": "goods_category_id",
        "other_fields": [],
        "conditional_fields": {
            "pos": {"goods_status": "1"},
            "neg": {}
        },
        "table": "amazon_category_train",
        "size": "",
        'partition_id': 'ID'
    }

    mysql_conf = {"user": "test", 'password': 'test', 'host': '127.0.0.1', 'db': 'test', 'charset': 'utf8'}

    WORDS_OUTPUT_COL = "words"
    STRING_INDEX_COL = "indexed"
    VECTOR_INDEX_COL = "vecmodel"

    VEC_PATH = "w2v_models/vec_model"

    INDEX_PATH = "w2v_models/index_model"

    CLF_PATH = "w2v_models/clf_model"

    DO_SAVE = False

    NUM_FEATURES = 2 ** 18

    PARTITIONS = 40


    def train_clf(df, vecModel):
        indexizer = text.Indexizer().get(inputCol="label", outputCol=STRING_INDEX_COL)
        clf = classifier.Classifier("ovr").get(featuresCol=VECTOR_INDEX_COL, labelCol=STRING_INDEX_COL
                                               , predictionCol='prediction', regParam=0.35)
        vecDF = vecModel.transform(df)
        idxModel = indexizer.fit(df)
        idxDF = idxModel.transform(vecDF)
        clfModel = clf.fit(idxDF)
        return idxModel, clfModel

    def predict(df, vecModel, idxModel,clfModel):
        vecDF = vecModel.transform(df)
        idxDF = idxModel.transform(vecDF)
        predDF = clfModel.transform(idxDF)
        return predDF


    def save_model(idxModel,clfModel):
        idxModel.save(INDEX_PATH)
        clfModel.save(CLF_PATH)



    spark = SparkSession.builder.appName("ml_category").getOrCreate()

    spark.read.jdbc(url="jdbc:mysql://" + mysql_conf['host'] + "/" + mysql_conf['db'], table=sql_info['table'],
                    column=sql_info['partition_id'], lowerBound=1000, upperBound=10000, numPartitions=PARTITIONS,
                    properties={"user": mysql_conf['user'], "password": mysql_conf['password']
                        , 'driver': "com.mysql.jdbc.Driver"}).createOrReplaceTempView(sql_info['table'])

    trainDF = spark.sql(render.select_sql(sql_info))

    tokenizer = text.Tokenizer(sql_info['text_fields'], 'remove')

    tokenDF = tokenizer.process(spark, trainDF, WORDS_OUTPUT_COL, sql_info['unique_id'])

    tokenTrainDF, tokenTestDF = tokenDF.randomSplit([0.7,0.3])

    vecModel = text.Vectorizer('w2v').get_model(VEC_PATH)

    idxModel, clfModel = train_clf(tokenTrainDF,vecModel)

    if DO_SAVE: save_model(idxModel,clfModel)

    #test data

    testResDF = predict(tokenTestDF,vecModel,idxModel,clfModel)

    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol=STRING_INDEX_COL)

    print(str(evaluator.evaluate(testResDF)))