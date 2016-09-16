
from pyspark.sql import SparkSession
from spark_libs import render,text, classifier
import pymysql

if __name__=="__main__":

    sql_info = {
        "unique_id": "seller_product_uuid",
        "text_fields":["seller_product_title","seller_product_current_description"],
        "label_field":"",
        "other_fields": [],
        "conditional_fields":{
            "pos":{"seller_product_status":"1", "seller_product_category":""},
            "neg":{}
        },
        "table":"seller_product",
        'partition_id': 'partition_id',
        "size":""
    }

    OUTPUT_COL = "ml_product_category"

    update_info = {
        "update_fields":{
            "seller_product_category" : OUTPUT_COL,
            "seller_product_category_probability": "highest_prob"
        },
        "conditional_fields":{
            "pos":{"seller_product_uuid":"seller_product_uuid"},
            "neg": {}
        },
        "table":"seller_product"
    }


    mysql_conf = {"user": "test", 'password': 'test', 'host': '127.0.0.1', 'db': 'test', 'charset': 'utf8'}

    WORDS_OUTPUT_COL = "words"

    VEC_PATH = "ml_models/vec_model"

    INDEX_PATH = "ml_models/index_model"

    CLF_PATH = "ml_models/clf_model"

    def send_data(data):
        sql = render.update_prod_sql(update_info)
        conn = pymysql.connect(**mysql_conf)
        cursor = conn.cursor()
        for d in data:
            update_value = [d[field] for field in update_info["update_fields"].values()]
            update_value.extend([d[field] for field in update_info['conditional_fields']['pos'].values()])
            update_value.extend([d[field] for field in update_info['conditional_fields']['neg'].values()])
            cursor.execute(sql,tuple(update_value))
            conn.commit()
        conn.close()

    def predict_clf(df):
        vectorizer = text.Vectorizer('tfidf').get_model(VEC_PATH)
        indexizer = text.Indexizer().get_model(INDEX_PATH,outputCol=OUTPUT_COL)
        clf = classifier.Classifier("nb").get_model(CLF_PATH)
        vecDF = vectorizer.transform(df)
        clfDF = clf.transform(vecDF)
        idxDF = indexizer.transform(clfDF)
        return idxDF



    spark = SparkSession.builder.appName("ml_predict").getOrCreate()

    spark.read.jdbc(url="jdbc:mysql://" + mysql_conf['host'] + "/" + mysql_conf['db'], table=sql_info['table'],
                    column=sql_info['partition_id'], lowerBound=1000, upperBound=10000, numPartitions=40,
                    properties={"user": mysql_conf['user'], "password": mysql_conf['password']
                    , 'driver':"com.mysql.jdbc.Driver"}).createOrReplaceTempView(sql_info['table'])

    predDF = spark.sql(render.select_sql(sql_info))

    tokenizer = text.Tokenizer(sql_info['text_fields'], 'remove')

    tokenDF = tokenizer.process(spark, predDF, WORDS_OUTPUT_COL, sql_info['unique_id'])

    resDF = predict_clf(tokenDF)

    probDF = spark.createDataFrame(resDF.rdd.map(lambda a: (a[sql_info['unique_id']],max(a['probability']).item()))
                                  ,[sql_info['unique_id'],'highest_prob'])

    resDF = resDF.join(probDF, sql_info['unique_id'],"inner").select(sql_info['unique_id'],OUTPUT_COL, 'highest_prob')

    resDF.foreachPartition(send_data)


