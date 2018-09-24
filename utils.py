def show_lda_weights(cvmodel, topics):

    for x, topic in enumerate(topics):
        print ('topic nr: ' + str(x))
        words = topic[0]
        weights = topic[1]
        for n in range(20):
            print (cvmodel.vocabulary[words[n]] + ' ' + str(weights[n]))