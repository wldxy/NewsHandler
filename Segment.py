from nltk.corpus import stopwords
import nltk
import collections
import string
from gensim import corpora

def segment(docs, dicname="dict.txt", vecname="vec.txt"):
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
    dic = corpora.Dictionary(tokens)
    dic.save(dicname)
    corpus = [dic.doc2bow(token) for token in tokens]
    corpora.MmCorpus.serialize(vecname, corpus)

    return corpus