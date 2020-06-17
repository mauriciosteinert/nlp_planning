from utils import WordEmbedding
from wikihow import Wikihow
import numpy as np
import nltk

class ActionIdentifier():
    def __init__(self, config):
        self.config = config
        self.word_embedding = WordEmbedding.WordEmbedding(config)


    def run(self):
        print("Performing action identifier experiment ...")
        # Create dataset object
        wikihow = Wikihow.Wikihow(self.config)
        instance = wikihow.process_example(wikihow.get_entry(0))

        for sentence in instance:
            print("Sentence: {}".format(sentence))
            # Tokenize
            sentence_tokens = nltk.word_tokenize(sentence)
            sentence_tags = nltk.pos_tag(sentence_tokens)
            print("Tokens: {}".format(sentence_tags))

            for token, tag in zip(sentence_tokens, sentence_tags):
                keyword_similarity = []
                for keyword in self.config['action_identifier']['keywords']:
                    d = 1.0 - self.word_embedding.get_distance(token, keyword)[2]
                    keyword_similarity.append(d)

                sentence_entry = (token, tag, self.word_embedding.get_word_vector(token), keyword_similarity, np.mean(keyword_similarity))
                print(sentence_entry)


        # Convert sentences to embedding
        # sentences_embeddings = self.word_embedding.get_sentence_list_embedding(instance)
        # keyword_embeddings = self.word_embedding.get_sentence_list_embedding(self.config['action_identifier']['keywords'])
        # sentences_tokens = []
        # distance = []
        #
        # for sentence in instance:
        #     sentences_tokens.append(nltk.pos_tag(nltk.word_tokenize(sentence)))
        #
        #
        # for sentence_list in sentences_embeddings:
        #     for sentence, sentence_tokens in zip(sentence_list, sentences_tokens):
        #         # Compute similarity of each word to predefined keywords
        #         for word, token in zip(sentence, sentence_tokens):
        #             distance_keyword = []
        #             for keyword in keyword_embeddings:
        #                 if len(word[1]) == 1:
        #                     distance_keyword.append(float('inf'))
        #                 else:
        #                     d = 1.0 - self.word_embedding.model.distance(keyword[0][0][0], word[0])
        #                     distance_keyword.append(d)
        #
        #             distance.append((word[0], token, distance_keyword, np.mean(distance_keyword)))
        #
        # for d in distance:
        #     print(d)
        #     print()

        # print("Keywords:")
        # for keyword in keyword_embeddings:
        #     print(keyword)



        return None
