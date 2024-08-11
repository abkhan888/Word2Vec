from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

reviews=[
    'the product is very good and the quality is fantastic',
    'i love the services and the product was satisfactory',
    'product that i have ordered came broken and customer services are very poor',
    'great product love he quality and customer services'
]
sentences = [simple_preprocess(review) for review in reviews]

model=Word2Vec(sentences,vector_size=100, window=5, min_count=1, sg=0)
vector = model.wv['product']
print(vector)
similar_words = model.wv.most_similar('product')
print(similar_words)