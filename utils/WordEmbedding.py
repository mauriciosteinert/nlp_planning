from utils import utils
from gensim.models import KeyedVectors
import nltk


class WordEmbedding:
    def __init__(self, config):
        self.model = KeyedVectors.load_word2vec_format(config['word_embedding_dict'],
                                                        binary=True,
                                                        unicode_errors='ignore')


    def get_word_vector(self, word):
        try:
            return (word, self.model.get_vector(word))
        except KeyError:
            return (word, [0])


    def get_distance(self, word1, word2):
        return (word1, word2, self.model.distance(word1, word2))


    def get_sentence_embedding(self, sentence):
        # words = sentence.split(' ')
        words = nltk.word_tokenize(sentence)
        sentence_embedding = []
        for word in words:
            try:
                sentence_embedding.append((word, self.model.get_vector(word)))
            except KeyError:
                sentence_embedding.append((word, [0]))

        return sentence_embedding


    def get_sentence_list_embedding(self, list_sentence):
        list_embeddings = []
        for sentence in list_sentence:
            list_embeddings.append([self.get_sentence_embedding(sentence)])

        return list_embeddings
