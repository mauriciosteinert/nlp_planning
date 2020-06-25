from utils import WordEmbedding
from utils import utils
from wikihow import Wikihow
import numpy as np
import nltk
from nltk.corpus import verbnet
import spacy

import os
from tqdm import trange
import time


class ActionIdentifier():
    def __init__(self, config):
        self.config = config
        self.word_embedding = WordEmbedding.WordEmbedding(config)


    def run(self):
        print("Performing action identifier experiment ...")
        open(self.config['log_file'], 'w')
        count = 0
        sentences_total = 0
        start_time = time.time()

        utils.write_log(self.config, "RUNNING CONFIGURATION: {}".format(self.config))

        # Create dataset object
        wikihow = Wikihow.Wikihow(self.config)

        statistic_list = []
        statistic_similarity = []
        ground_truth_count = 0
        dataset_length = int(wikihow.get_length() * self.config['action_identifier']['dataset_evaluation_percent'])

        if dataset_length < 1:
            print("No examples to process in dataset. Aborting ...")
            return


        for idx in trange(dataset_length):
            instance = wikihow.get_entry(idx)
            text = wikihow.process_example(instance[1])
            utils.write_log(self.config, "\n---------------------------------------------------------------------------\n")
            utils.write_log(self.config, "FILE: {}\n".format(instance[0]))
            spacy_en = spacy.load('en_core_web_sm')

            for sentence in text:
                sentences_total += 1

                # Tokenize
                if self.config['action_identifier']['ground_truth_generator'] == 'nltk':
                    sentence_tokens = nltk.word_tokenize(sentence)
                    sentence_tags = nltk.pos_tag(sentence_tokens)
                    ground_truth_verbs = [v[0] for v in sentence_tags if len(verbnet.classids(v[0])) > 0]
                elif self.config['action_identifier']['ground_truth_generator'] == 'spacy':
                    doc = spacy_en(sentence)
                    sentence_tokens = [t for t in doc]
                    sentence_tags = [(str(t), t.pos_) for t in doc]
                    ground_truth_verbs = [v for v in doc if v.pos_ == 'VERB']
                else:
                    print("No ground-truth mechanism defined! Aborting ...")
                    return

                utils.write_log(self.config, "\n>SENTENCE: {}".format(sentence))
                utils.write_log(self.config, "\n  >SENTENCE TAGS: {}".format(sentence_tags))

                if len(ground_truth_verbs) == 0:
                    ground_truth_count += 1

                utils.write_log(self.config, "\n  >GROUND-TRUTH VERBS: {}".format(ground_truth_verbs))

                embedding_verbs = []

                for token, tag in zip(sentence_tokens, sentence_tags):
                    keyword_similarity = []
                    for keyword in self.config['action_identifier']['keywords']:
                        try:
                            similarity = 1.0 - self.word_embedding.get_distance(str(token), str(keyword))[2]
                        except KeyError:
                            similarity = 0.0


                        keyword_similarity.append(similarity)

                    mean = np.mean(keyword_similarity)

                    if mean >= float(self.config['action_identifier']['similarity_threshold']):
                        embedding_verbs.append((token, mean))
                        statistic_similarity.append(mean)

                true_positive = [e[0] in ground_truth_verbs for e in embedding_verbs]
                true_positive = np.count_nonzero(true_positive)

                false_positive = [e[0] not in ground_truth_verbs for e in embedding_verbs]
                false_positive = np.count_nonzero(false_positive)

                embedding_verbs_words = [str(e[0]) for e in embedding_verbs]
                ground_truth_verbs = [str(e) for e in ground_truth_verbs]

                false_negative = [e not in embedding_verbs_words for e in ground_truth_verbs]
                false_negative = np.count_nonzero(false_negative)

                sentence_entry = (token, tag, self.word_embedding.get_word_vector(token), keyword_similarity, mean)

                utils.write_log(self.config, "\n  >EMBEDDING VERBS: {}".format(embedding_verbs))

                # Text statistics [true positive, false negative, precision, recall, f-score]
                try:
                    precision = true_positive / (true_positive + false_positive)
                except ZeroDivisionError:
                    precision = 0.0

                try:
                    recall = true_positive / (true_positive + false_negative)
                except ZeroDivisionError:
                    recall = 0.0

                try:
                    f_score = 2 * (recall * precision) / (recall + precision)
                except ZeroDivisionError:
                    f_score = 0.0

                utils.write_log(self.config, "\n  >TP: {} FP: {} FN: {} Precision: {} Recall: {} F-Score: {}".format(true_positive, false_positive, false_negative, precision, recall, f_score))
                statistic_list.append([true_positive, false_positive, false_negative, precision, recall, f_score])
            count += 1

        print("Calculating statistics ...")
        statistic_mean = np.mean(statistic_list, axis=0)
        statistic_std = np.std(statistic_list, axis=0)

        utils.write_log(self.config, "\n=======================================================================\n")
        utils.write_log(self.config, "RESULTS (Elapsed time: {:.4f} seconds)".format(time.time() - start_time))
        utils.write_log(self.config, "\n  Total of examples: {}".format(count))
        utils.write_log(self.config, "\n  Total of sentences: {} - Mean per example: {:.4f} - Ground-truth sentences with zero verbs: {} ({:.4f} %)".format(sentences_total, \
                            sentences_total / count, ground_truth_count, ground_truth_count / sentences_total))
        utils.write_log(self.config, "\n  Mean True Positive: {:.4f} - Std: {:.4f}".format(statistic_mean[0], statistic_std[0]))
        utils.write_log(self.config, "\n  Mean False Positive: {:.4f} - Std: {:.4f}".format(statistic_mean[1], statistic_std[1]))
        utils.write_log(self.config, "\n  Mean False Negative: {:.4f} - Std: {:.4f}".format(statistic_mean[2], statistic_std[2]))
        utils.write_log(self.config, "\n  Mean Similarity: {:.4f} - Std: {:.4f}".format(np.mean(statistic_similarity), np.std(statistic_similarity)))
        utils.write_log(self.config, "\n  Mean Precision: {:.4f} - Recall: {:.4f} - F-Score: {:.4f}".format(
            statistic_mean[3], statistic_mean[4], statistic_mean[5]))
