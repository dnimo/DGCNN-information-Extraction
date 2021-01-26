import unittest


from gensim.models import KeyedVectors

# with open('./../data/financial.word.txt') as fin:
#     print(Word2Vec.load(fin))

KeyedVectors.load_word2vec_format('./../data/financial.word.txt')

if __name__ == '__main__':
    unittest.main()
