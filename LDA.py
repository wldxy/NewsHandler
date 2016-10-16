from nltk.corpus import stopwords
import nltk
import collections
import string
from gensim import corpora, models

class LDA():
    ready = False

    def __init__(self):
        pass

    def segment(self, docs):
        stopword = stopwords.words('english')
        tokens = list()
        punc = string.punctuation
        for doc in docs:
            tokens.append(nltk.word_tokenize(doc))
        frequency = collections.defaultdict(int)
        for token in tokens:
            for word in token:
                frequency[word] += 1

        tokens = [[word for word in token if frequency[word] > 1 and word not in stopword and word not in punc]
              for token in tokens]
        self.dic = corpora.Dictionary(tokens)
        self.corpus = [self.dic.doc2bow(token) for token in tokens]
        self.ready = True

    def ldatrain(self, num_topics=100):
        if not self.ready:
            return
        self.result = models.LdaModel(self.corpus, id2word=self.dic, num_topics=num_topics)

    def save(self, dictName="lda.dict", corpusName="lda.mm"):
        if not self.ready:
            return
        self.dic.save(dictName)
        corpora.MmCorpus.serialize(corpusName, self.corpus)

    def read(self, dictName, corpusName):
        self.dic = corpora.Dictionary.load(dictName)
        self.corpus = corpora.MmCorpus(corpusName)
        self.ready = True
