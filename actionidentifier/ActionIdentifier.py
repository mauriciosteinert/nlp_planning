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

        print("Length: {}".format(int(wikihow.get_length() * self.config['action_identifier']['dataset_evaluation_percent'])))
        statistic_list = []

        for idx in range(int(wikihow.get_length() * self.config['action_identifier']['dataset_evaluation_percent'])):
            instance = wikihow.get_entry(idx)
            text = wikihow.process_example(instance[1])
            utils.write_log(self.config, "\n---------------------------------------------------------------------------\n")
            utils.write_log(self.config, "FILE: {}\n".format(instance[0]))

            for sentence in text:
                # Tokenize
                sentence_tokens = nltk.word_tokenize(sentence)

                if self.config['action_identifier']['add_pronoun']:
                    sentence_tokens.insert(0, 'they')

                sentence_tags = nltk.pos_tag(sentence_tokens)

                utils.write_log(self.config, "\n>SENTENCE: {}".format(sentence))
                utils.write_log(self.config, "\n  >NLTK TAGS: {}".format(sentence_tags))
                nltk_verbs = [v[0] for v in sentence_tags if v[1] in ['VB', 'VBZ', 'VBP', 'VBG']]
                utils.write_log(self.config, "\n  >NLTK VERBS: {}".format(nltk_verbs))

                embedding_verbs = []
                mean_distance = []

                for token, tag in zip(sentence_tokens, sentence_tags):
                    keyword_similarity = []
                    for keyword in self.config['action_identifier']['keywords']:
                        try:
                            similarity = 1.0 - self.word_embedding.get_distance(token, keyword)[2]
                        except KeyError:
                            similarity = 0.0

                        keyword_similarity.append(similarity)

                    mean = np.mean(keyword_similarity)

                    if mean >= float(self.config['action_identifier']['similarity_threshold']):
                        embedding_verbs.append((token, mean))

                true_positive = [e[0] in nltk_verbs for e in embedding_verbs]
                try:
                    true_positive = np.count_nonzero(true_positive) / len(true_positive)
                except ZeroDivisionError:
                    true_positive = 0.0

                sentence_entry = (token, tag, self.word_embedding.get_word_vector(token), keyword_similarity, mean)

                utils.write_log(self.config, "\n  >EMBEDDING VERBS: {} - {}".format(embedding_verbs, true_positive))

            # Text statistics [true positive, false negative, mean_distance]
            statistic_list.append([true_positive, 0, 0])

        statistic_mean = np.mean(statistic_list, axis=0)
        utils.write_log(self.config, "\n=======================================================================\n")
        utils.write_log(self.config, "RESULTS")
        utils.write_log(self.config, "\nTotal of examples: {}".format(int(wikihow.get_length() * self.config['action_identifier']['dataset_evaluation_percent'])))
        utils.write_log(self.config, "\n  Mean True Positive: {}".format(statistic_mean[0]))
