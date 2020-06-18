from utils import WordEmbedding
from utils import utils
from wikihow import Wikihow
import numpy as np
import nltk
import os
from tqdm import tqdm


class ActionIdentifier():
    def __init__(self, config):
        self.config = config
        self.word_embedding = WordEmbedding.WordEmbedding(config)


    def run(self):
        print("Performing action identifier experiment ...")
        open(self.config['log_file'], 'w')
        count = 0

        utils.write_log(self.config, "RUNNING CONFIGURATION: {}".format(self.config))

        # Create dataset object
        wikihow = Wikihow.Wikihow(self.config)
        with tqdm(total=int(wikihow.get_length() * self.config['action_identifier']['dataset_evaluation_percent'])) as pbar:
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
                    statistic_similarity = []

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
                            statistic_similarity.append(mean)

                    true_positive = [e[0] in nltk_verbs for e in embedding_verbs]
                    try:
                        true_positive = np.count_nonzero(true_positive) / len(true_positive)
                    except ZeroDivisionError:
                        true_positive = 0.0

                    false_positive = [e[0] not in nltk_verbs for e in embedding_verbs]

                    try:
                        false_positive = np.count_nonzero(false_positive) / len(false_positive)
                    except ZeroDivisionError:
                        false_positive = 0.0

                    sentence_entry = (token, tag, self.word_embedding.get_word_vector(token), keyword_similarity, mean)

                    utils.write_log(self.config, "\n  >EMBEDDING VERBS: {} TP: {} FP: {}".format(embedding_verbs, true_positive, false_positive))

                # Text statistics [true positive, false negative, mean_distance]
                statistic_list.append([true_positive, false_positive, 0])
                pbar.update(idx)
                count += 1

        statistic_mean = np.mean(statistic_list, axis=0)
        statistic_std = np.std(statistic_list, axis=0)

        utils.write_log(self.config, "\n=======================================================================\n")
        utils.write_log(self.config, "RESULTS")
        utils.write_log(self.config, "\nTotal of examples: {}".format(count))
        utils.write_log(self.config, "\n  Mean True Positive: {} Std: {}".format(statistic_mean[0], statistic_std[0]))
        utils.write_log(self.config, "\n  Mean False Positive: {} Std: {}".format(statistic_mean[1], statistic_std[1]))
        utils.write_log(self.config, "\n  Mean Similarity: {} Std: {}".format(np.mean(statistic_similarity), np.std(statistic_similarity)))
