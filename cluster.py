#coding:utf-8

from gensim import corpora, models, similarities, matutils, utils
import cPickle
import logging
import os
import numpy as np
import scipy
from collections import defaultdict
import sys

data_dir = os.path.join(os.getcwd(), 'data')
# work_dir = os.path.join(os.getcwd(), 'model', os.path.basename(__file__).rstrip('.py'))
# if not os.path.exists(work_dir):
#     os.mkdir(work_dir)
# os.chdir(work_dir)

logger = logging.getLogger('text_similar')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# convert to unicode
def to_unicode(text):
    if not isinstance(text, unicode):
        text = text.decode('utf-8')
    return text

class TextSimilar(utils.SaveLoad):
    def __init__(self):
        self.conf = {}

    def _preprocess(self):
        docs = [to_unicode(doc.strip()).split()[1:] for doc in file(self.fname)]
        cPickle.dump(docs, open(self.conf['fname_docs'], 'wb'))

        dictionary = corpora.Dictionary(docs)
        dictionary.save(self.conf['fname_dict'])

        corpus = [dictionary.doc2bow(doc) for doc in docs]
        corpora.MmCorpus.serialize(self.conf['fname_corpus'], corpus)

        return docs, dictionary, corpus

    def _generate_conf(self):
        fname = self.fname[self.fname.rfind('/') + 1:]
        self.conf['fname_docs']   = '%s.docs' % fname
        self.conf['fname_dict']   = '%s.dict' % fname
        self.conf['fname_corpus'] = '%s.mm' % fname

    def train(self, fname, is_pre=True, method='lsi', **params):
        self.fname = fname
        self.method = method
        self._generate_conf()
        if is_pre:
            self.docs, self.dictionary, corpus = self._preprocess()
        else:
            self.docs = cPickle.load(open(self.conf['fname_docs']))
            self.dictionary = corpora.Dictionary.load(self.conf['fname_dict'])
            corpus = corpora.MmCorpus(self.conf['fname_corpus'])

        if params is None:
            params = {}

        logger.info("training TF-IDF model")
        self.tfidf = models.TfidfModel(corpus, id2word=self.dictionary)
        corpus_tfidf = self.tfidf[corpus]

        if method == 'lsi':
            logger.info("training LSI model")
            self.lsi = models.LsiModel(corpus_tfidf, id2word=self.dictionary, **params)
            self.similar_index = similarities.MatrixSimilarity(self.lsi[corpus_tfidf])
            self.para = self.lsi[corpus_tfidf]
        elif method == 'lda_tfidf':
            logger.info("training LDA model")
            self.lda = models.LdaMulticore(corpus_tfidf, id2word=self.dictionary, workers=8, **params)
            self.similar_index = similarities.MatrixSimilarity(self.lda[corpus_tfidf])
            self.para = self.lda[corpus_tfidf]
        elif method == 'lda':
            logger.info("training LDA model")
            self.lda = models.LdaMulticore(corpus, id2word=self.dictionary, workers=8, **params)
            self.similar_index = similarities.MatrixSimilarity(self.lda[corpus])
            self.para = self.lda[corpus]
        elif method == 'logentropy':
            logger.info("training a log-entropy model")
            self.logent = models.LogEntropyModel(corpus, id2word=self.dictionary)
            self.similar_index = similarities.MatrixSimilarity(self.logent[corpus])
            self.para = self.logent[corpus]
        else:
            msg = "unknown semantic method %s" % method
            logger.error(msg)
            raise NotImplementedError(msg)

    def doc2vec(self, doc):
        bow = self.dictionary.doc2bow(to_unicode(doc).split())
        if self.method == 'lsi':
            return self.lsi[self.tfidf[bow]]
        elif self.method == 'lda':
            return self.lda[bow]
        elif self.method == 'lda_tfidf':
            return self.lda[self.tfidf[bow]]
        elif self.method == 'logentropy':
            return self.logent[bow]

    def find_similar(self, doc, n=10):
        vec = self.doc2vec(doc)
        sims = self.similar_index[vec]
        sims = sorted(enumerate(sims), key=lambda item: -item[1])
        for elem in sims[:n]:
            idx, value = elem
            print ' '.join(self.docs[idx]), value

    def get_vectors(self):
        return self._get_vector(self.para)

    def _get_vector(self, corpus):

        def get_max_id():
            maxid = -1
            for document in corpus:
                maxid = max(maxid, max([-1] + [fieldid for fieldid, _ in document])) # [-1] to avoid exceptions from max(empty)
            return maxid

        num_features = 1 + get_max_id()
        index = np.empty(shape=(len(corpus), num_features), dtype=np.float32)
        for docno, vector in enumerate(corpus):
            if docno % 1000 == 0:
                print("PROGRESS: at document #%i/%i" % (docno, len(corpus)))

            if isinstance(vector, np.ndarray):
                pass
            elif scipy.sparse.issparse(vector):
                vector = vector.toarray().flatten()
            else:
                vector = matutils.unitvec(matutils.sparse2full(vector, num_features))
            index[docno] = vector

        print index.shape

        return index


def cluster(vectors, ts, k=30):
    from sklearn.cluster import k_means
    X = np.array(vectors)
    cluster_center, result, inertia = k_means(X.astype(np.float), n_clusters=k, init="k-means++")
    X_Y_dic = defaultdict(set)
    for i, pred_y in enumerate(result):
        X_Y_dic[pred_y].add(' '.join(ts.docs[i]))

    print 'len(X_Y_dic): ', len(X_Y_dic)
    with open(data_dir + '/cluser.txt', 'w') as fo:
        for Y in X_Y_dic:
            fo.write(str(Y) + '\n')
            fo.write('{word}\n'.format(word='\n'.join(list(X_Y_dic[Y])[:100]).encode('utf-8')))

def main(is_train=True):
    fname = data_dir + '/cuisine.txt'

    num_topics = 100
    method = 'lda'

    ts = TextSimilar()
    if is_train:
        ts.train(fname, method=method ,num_topics=num_topics, is_pre=True, iterations=100)
        ts.save(method)
    else:   
        ts = TextSimilar().load(method)

    index = ts.get_vectors()
    cluster(index, ts, k=num_topics)

if __name__ == '__main__':
    is_train = True if len(sys.argv) > 1 else False
    main(is_train)