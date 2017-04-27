import math
import random
import logging
import numpy
import glob
import argparse
from gensim import models, matutils, corpora, similarities
from sklearn.feature_extraction.text import TfidfVectorizer
from time import time
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity


def sim_matrix():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=10000,
                                     min_df=2, stop_words='english',
                                     use_idf=True)
    text = []
    c_names = []
    cat_list = glob.glob ("categories/*")
    cat_size = len(cat_list)
    if cat_size < 1:
        print "you need to generate the cuisines files 'categories' folder first"
        return

    sample_size = min(30, cat_size)
    cat_sample = sorted( random.sample(range(cat_size), sample_size) )
    #print (cat_sample)
    count = 0
    for i, item in enumerate(cat_list):
        if i == cat_sample[count]:
            li =  item.split('/')
            cuisine_name = li[-1]
            c_names.append(cuisine_name[:-4].replace("_"," "))
            with open ( item ) as f:
                text.append(f.read().replace("\n", " "))
            count = count + 1
        
        if count >= len(cat_sample):
            print "generating cuisine matrix with:", count, "cuisines"
            break

    print (c_names)

    X = vectorizer.fit_transform(text)
    sim = cosine_similarity(X)
    # print sim
    # numpy.savetxt("withidf.csv", sim, delimiter=",")

    id2words ={}
    for i,word in enumerate(vectorizer.get_feature_names()):
        id2words[i] = word

    corpus = matutils.Sparse2Corpus(X, documents_columns=False)
    lda = models.ldamodel.LdaModel(corpus, num_topics=100, passes=2, id2word=id2words)
    lda_clus = models.ldamodel.LdaModel(corpus, num_topics=3, passes=2, id2word=id2words)

    doc_topics = lda_clus.get_document_topics(corpus)
    cuisine_matrix = []

    for i, doc_a in enumerate(doc_topics):
        #print (i)
        sim_vecs = []
        for j , doc_b in enumerate(doc_topics):
            w_sum = 0
            if ( i <= j ):
                norm_a = 0
                norm_b = 0
                
                for (my_topic_b, weight_b) in doc_b:
                    norm_b = norm_b + weight_b*weight_b

                for (my_topic_a, weight_a) in doc_a:
                    norm_a = norm_a + weight_a*weight_a
                    for (my_topic_b, weight_b) in doc_b:
                        if ( my_topic_a == my_topic_b ):
                            w_sum = w_sum + weight_a*weight_b

                norm_a = math.sqrt(norm_a)
                norm_b = math.sqrt(norm_b)
                denom = (float) (norm_a * norm_b)
                if denom < 0.0001:
                    w_sum = 0
                else:
                    w_sum = w_sum/(denom)
            else:
                w_sum = cuisine_matrix[j][i]
            sim_vecs.append(w_sum)

        cuisine_matrix.append(sim_vecs)

    # similar_index = similarities.MatrixSimilarity(lda[corpus])
    # lda.print_topics(20)

    with open( 'lda_similarity_matrix.csv', 'w') as f:
        for i_list in cuisine_matrix:
            s = ""
            for tt in i_list:
                s = s+str(tt) + " "
            s = s.strip()
            f.write(",".join(s.split())+"\n")

    # with open('cuisine_indices.txt', 'w') as f:
    #     f.write( "\n".join(c_names))


if __name__=="__main__":
	sim_matrix()
