from gensim.models import word2vec

model = word2vec.Word2Vec.load("word2vec.model")

print(model.wv['high'])