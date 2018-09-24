from config import LDA_CLUSTERS
from pyspark.ml.clustering import LDA
from pyspark.ml.clustering import BisectingKMeans
from utils import show_lda_weights


def lda_model(data):

    lda = LDA(k=LDA_CLUSTERS,
              # seed=123,
              # optimizer="em",
              featuresCol="vectors")

    # todo Gridsearch best parameters

    model = lda.fit(data)
    topics = model.describeTopics(maxTermsPerTopic=15)
    print("Learned topics (as distributions over vocab of " + str(model.vocabSize())
          + " words):")
    wordNumbers = 10
    topicIndices = model.describeTopics(maxTermsPerTopic=wordNumbers)
    topicIndices.show()
    #does not work as it shown in dics. Seems to be in process in current Python API
    show_lda_weights(model, topics)


def bisect_model(data):
    #TODO grid search best parametrs
    bkm = BisectingKMeans().setK(2).setSeed(1)
    model = bkm.fit(data)
    cost = model.computeCost(data)
    print("Within Set Sum of Squared Errors = " + str(cost))
    print("Cluster Centers: ")
    centers = model.clusterCenters()
    for center in centers:
        print(center)
    predictions_bi = model.transform(data)

    return predictions_bi


