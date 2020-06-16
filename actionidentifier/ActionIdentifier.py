from utils import WordEmbedding
from wikihow import Wikihow

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
        embeddings = self.word_embedding.get_sentence_list_embedding(instance)

        for sentence_list in embeddings:
            for sentence in sentence_list:
                # Compute distance of each word to predefined keywords
                for word in sentence:
                    # TODO
                    print(word[0])
            print()



        return None
