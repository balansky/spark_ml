from pyspark.sql import SparkSession
from spark_libs import text, render

if __name__ == "__main__":
    sql_info = {
        "unique_id": "goods_uuid",
        "text_fields": ["goods_title", "goods_description"],
        "label_field": "goods_category_id",
        "other_fields": [],
        "conditional_fields": {
            "pos": {},
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

    DO_SAVE = True

    NUM_FEATURES = 400

    PARTITIONS = 40


    spark = SparkSession.builder.appName("ml_category").getOrCreate()

    spark.read.jdbc(url="jdbc:mysql://" + mysql_conf['host'] + "/" + mysql_conf['db'], table=sql_info['table'],
                    column=sql_info['partition_id'], lowerBound=1000, upperBound=10000, numPartitions=PARTITIONS,
                    properties={"user": mysql_conf['user'], "password": mysql_conf['password']
                        , 'driver': "com.mysql.jdbc.Driver"}).createOrReplaceTempView(sql_info['table'])

    trainDF = spark.sql(render.select_sql(sql_info))

    tokenizer = text.Tokenizer(sql_info['text_fields'], 'remove')

    tokenDF = tokenizer.process(spark,trainDF,WORDS_OUTPUT_COL, sql_info['unique_id'])


    vectorizer = text.Vectorizer('w2v').get(vectorSize=NUM_FEATURES, inputCol=WORDS_OUTPUT_COL
                                              ,outputCol=VECTOR_INDEX_COL, numPartitions=PARTITIONS)
    vecModel = vectorizer.fit(tokenDF)

    vecModel.save(VEC_PATH)

    resDF = vecModel.transform(tokenDF)

    resDF.show()