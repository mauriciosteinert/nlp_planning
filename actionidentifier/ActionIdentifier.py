from utils import WordEmbedding
from utils import utils
from wikihow import Wikihow
import numpy as np
import nltk
import os


class ActionIdentifier():
    def __init__(self, config):
        self.config = config
        self.word_embedding = WordEmbedding.WordEmbedding(config)


    def run(self):
        print("Performing action identifier experiment ...")
        open(self.config['log_file'], 'w')

        # Create dataset object
        wikihow = Wikihow.Wikihow(self.config)

        for idx in range(2):
            instance = wikihow.get_entry(idx)
            text = wikihow.process_example(instance[1])
            utils.write_log(self.config, "\n---------------------------------------------------------------------------\n")
            utils.write_log(self.config, "FILE: {}\n".format(instance[0]))

            for sentence in text:
                # Tokenize
                sentence_tokens = nltk.word_tokenize(sentence)
                sentence_tags = nltk.pos_tag(sentence_tokens)

                utils.write_log(self.config, "\n>SENTENCE: {}".format(sentence))
                utils.write_log(self.config, "\n  >NLTK TAGS: {}".format(sentence_tags))
                nltk_verbs = [v[0] for v in sentence_tags if v[1] in ['VB', 'VBZ', 'VBP', 'VBG']]
                utils.write_log(self.config, "\n  >NLTK VERBS: {}".format(nltk_verbs))

                for token, tag in zip(sentence_tokens, sentence_tags):
                    embedding_verbs = []
                    keyword_similarity = []
                    for keyword in self.config['action_identifier']['keywords']:
                        d = 1.0 - self.word_embedding.get_distance(token, keyword)[2]
                        keyword_similarity.append(d)
                        mean = np.mean(keyword_similarity)

                    if mean >= self.config['action_identifier']['similarity_threshold']:
                        embedding_verbs.append((token, mean))

                    sentence_entry = (token, tag, self.word_embedding.get_word_vector(token), keyword_similarity, mean)
                utils.write_log(self.config, "\n  >EMBEDDING VERBS: {}".format(embedding_verbs))
                    # print(sentence_entry)


        return None
