from config import *
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, CountVectorizer, Tokenizer, StopWordsRemover, IDF
from pyspark.sql.functions import col


def data_preparation(path):

    spark = SparkSession \
        .builder \
        .master('local[*]') \
        .appName("Text_Classification") \
        .config("spark.executor.memory", "8g") \
        .getOrCreate()

    spark.sparkContext.setLogLevel('WARN')

    data = spark.read.csv([path])

    data = data.withColumnRenamed('_c0', 'owner')\
        .withColumnRenamed('_c1', 'id_comentator')\
        .withColumnRenamed('_c2', 'text')

    #todo test without drop
    drop_list = ['id_comentator']
    data = data.select([column for column in data.columns if column not in drop_list])

    data = data.na.drop(subset=["owner"])
    data = data.na.drop(subset=["text"])

    # todo implement Pipeline
    stringIndexer = StringIndexer(inputCol="owner", outputCol= "owner" + "Index")
    model = stringIndexer.fit(data)
    data = model.transform(data)
    #todo detect language (fasttext)

    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    wordsDataFrame = tokenizer.transform(data)

    cv_tmp = CountVectorizer(inputCol="words", outputCol="tmp_vectors")
    cv_tmp_model = cv_tmp.fit(wordsDataFrame)

    top20 = list(cv_tmp_model.vocabulary[0:20])
    more_then_3_charachters = [word for word in cv_tmp_model.vocabulary if len(word) <= 3]
    contains_digits = [word for word in cv_tmp_model.vocabulary if any(char.isdigit() for char in word)]
    stopwords = []
    stopwords = stopwords + top20 + more_then_3_charachters + contains_digits
    remover = StopWordsRemover(inputCol="words", outputCol="filtered", stopWords=stopwords)
    wordsDataFrame = remover.transform(wordsDataFrame)

    #TODO remove additionally stopwords after models run

    cv = CountVectorizer(inputCol="filtered", outputCol="vectors")
    cvmodel = cv.fit(wordsDataFrame)
    df_vect = cvmodel.transform(wordsDataFrame)

    idf = IDF(inputCol="vectors", outputCol="features")
    idfModel = idf.fit(df_vect)
    rescaledData = idfModel.transform(df_vect)
    rescaledData.select("ownerIndex", "features").show()

    return rescaledData