from utils import WordEmbedding
from wikihow import Wikihow
import numpy as np

class ActionIdentifier():
    def __init__(self, config):
        self.config = config
        self.word_embedding = WordEmbedding.WordEmbedding(config)


    def run(self):
        print("Performing action identifier experiment ...")
        # Create dataset object
        wikihow = Wikihow.Wikihow(self.config)
        instance = wikihow.process_example(wikihow.get_entry(0))

        print(instance)
        # Convert sentences to embedding
        sentences_embeddings = self.word_embedding.get_sentence_list_embedding(instance)
        keyword_embeddings = self.word_embedding.get_sentence_list_embedding(self.config['action_identifier']['keywords'])

        distance = []

        for sentence_list in sentences_embeddings:
            for sentence in sentence_list:
                # Compute distance of each word to predefined keywords
                for word in sentence:
                    distance_keyword = []
                    for keyword in keyword_embeddings:
                        if len(word[1]) == 1:
                            distance_keyword.append(float('inf'))
                        else:
                            d = np.linalg.norm((keyword[0][0][1], word[1]))
                            # print(word[0], keyword[0][0][0], d)
                            distance_keyword.append(d)
                    distance.append((word[0], distance_keyword))

        for d in distance:
            print(d)
            print()

        # print("Keywords:")
        # for keyword in keyword_embeddings:
        #     print(keyword)



        return None
