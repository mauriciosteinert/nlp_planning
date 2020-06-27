import torch
import spacy
import gensim
import os
import re
import numpy as np
from tqdm import trange
import time
import sent2vec

######################
# Parameters setup
torch.manual_seed(1)
WORD2VEC_DIM = 100
BATCH_SIZE = 1
TEST_SPLIT=0.2
EPOCHS = 50
CUT_POINT=2000
PYTORCH_MODEL_WEIGHTS_FILE = '/home/mauricio/repo/pucrs_nlp-planning/steps_checkpoint/model_weights_sent2vec.pt'
TRAIN_LOG='/home/mauricio/repo/pucrs_nlp-planning/steps_train_sent2vec.log'
TEST_LOG='/home/mauricio/repo/pucrs_nlp-planning/steps_test_sent2vec.log'



class WikihowDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, word2vec_dict, word2vec_dim):
        self.spacy_en = spacy.load('en')
        self.X = []
        self.Y = []
        self.max_sentences_in_text = 0
        self.model = sent2vec.Sent2vecModel()
        self.model.load_model('/home/mauricio/repo/datasets/word_vectors/enwiki_sent2vec_100.bin')

        self.files_list = os.listdir(os.path.abspath(dataset_dir))

        for idx, file in enumerate(self.files_list):
            print("{} Processing file {}".format(idx, file))
            sample = []
            with open(os.path.join(os.path.abspath(dataset_dir), file), 'r') as f:
                lines = f.read().split("\n")

                text = [l.split(".")[1].rstrip().lstrip().lower() for l in lines if re.match('^STEP.*', l)]
                sentences_count = len(text)

                if sentences_count > self.max_sentences_in_text:
                    self.max_sentences_in_text = sentences_count


            self.X.append(text)
            self.Y.append([1.0])
            self.X.append(text[::-1])
            self.Y.append([0.0])
        print("Maximum sentences in text: {}".format(self.max_sentences_in_text))


    def __len__(self):
        return len(self.X)


    def __getitem__(self, idx):
        res = self.model.embed_sentences(self.X[idx])
        res = torch.tensor(res).float()

        res_padding = torch.zeros(1, 100)
        if res.shape[0] != self.max_sentences_in_text:
            res_padding = torch.cat([res_padding] * (self.max_sentences_in_text - res.shape[0]))
            res = torch.cat((res, res_padding))

        return {'text_embedding': res.reshape(-1, 100), 'label': torch.tensor(self.Y[idx]).float()}



class NN(torch.nn.Module):
    def __init__(self, embed_size, hidden_size, num_layers, device):
        super(NN, self).__init__()
        self.device = device
        self.num_layers = num_layers
        self.hidden_size= hidden_size

        self.lstm = torch.nn.LSTM(input_size=embed_size,
                                  hidden_size=self.hidden_size,
                                  num_layers=self.num_layers,
                                  batch_first=True,
                                  bias=True)


        self.fc = torch.nn.Linear(self.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()


    def forward(self, x):
        out, _ = self.lstm(x, None)
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out





wikihow_dataset = WikihowDataset(dataset_dir="/home/mauricio/repo/datasets/wikihow/",
                                word2vec_dict="/home/mauricio/repo/datasets/word_vectors/enwiki_20200501_d100_small.bin",
#                                word2vec_dict="/home/mauricio/repo/datasets/word_vectors/enwiki_word2vec_20200501_d100_b.bin",
                                 word2vec_dim=WORD2VEC_DIM)

wikihow_dataloader = torch.utils.data.DataLoader(wikihow_dataset,
                                                 batch_size=BATCH_SIZE,
                                                 shuffle=True,
                                                 num_workers=2)


# Split dataset in train/validation/test
shuffle_dataset = True

dataset_size = len(wikihow_dataset)

indices = list(range(dataset_size))
split = int(np.floor(TEST_SPLIT * dataset_size))

if shuffle_dataset:
    np.random.seed(1234)
    np.random.shuffle(indices)

train_indices, test_indices = indices[split:], indices[:split]

train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)


train_loader = torch.utils.data.DataLoader(wikihow_dataset,
                                          batch_size=BATCH_SIZE,
                                          sampler=train_sampler,
                                          num_workers=4)

test_loader = torch.utils.data.DataLoader(wikihow_dataset,
                                          batch_size=BATCH_SIZE,
                                          sampler=test_sampler,
                                          num_workers=4)

print("Dataloader sizes: Train: {} Test: {}".format(len(train_loader), len(test_loader)))



torch.cuda.empty_cache()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("device: {}".format(device))
model = NN(WORD2VEC_DIM, 16, 2, device)

model.to(device)
optimizer = torch.optim.SGD(lr=0.01, params=model.parameters(), momentum=0.9)
criterion = torch.nn.BCELoss()




##########
# Train model
torch.manual_seed(1)

torch.autograd.set_detect_anomaly(True)
model.train()

print("Wikihow data length: {}".format(wikihow_dataset[0]['text_embedding'].shape))

with open(TRAIN_LOG, 'w') as f:
    for epoch in range(EPOCHS):
        loss_acc = 0
        start_time = time.time()
        for idx, batch in enumerate(train_loader):
            batch_size = batch['text_embedding'].size(0)
            x = batch['text_embedding'].reshape(-1)[:CUT_POINT].reshape(BATCH_SIZE, -1, WORD2VEC_DIM).to(device)
            # x = batch['text_embedding'].reshape(BATCH_SIZE, -1, WORD2VEC_DIM).to(device)
            y = batch['label'].to(device)

            y_hat = model(x)
            loss = criterion(y_hat.squeeze(), y.squeeze())
    #         print(y, y_hat, loss)
            loss_acc += loss
            model.zero_grad()
            loss.backward()
            optimizer.step()

        f.write("\n{} Loss: {} - elapsed time: {:.4f}".format(epoch, loss_acc, time.time() - start_time))
        print("{} Loss: {} - elapsed time: {:.4f}".format(epoch, loss_acc, time.time() - start_time))

##########
# Save model weights
# Save model weights
torch.save(model.state_dict(), PYTORCH_MODEL_WEIGHTS_FILE)



##########
# Validation
# Validate

model.eval()

positive_examples_count = 0
negative_examples_count = 0

true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0

with open(TEST_LOG, 'w') as f:
    for idx, batch in enumerate(test_loader):
        x = batch['text_embedding'].reshape(-1)[:CUT_POINT].reshape(1, -1, WORD2VEC_DIM).to(device)
        # x = batch['text_embedding'].reshape(1, -1, WORD2VEC_DIM).to(device)
        y = batch['label'].to(device)
        y_hat = model(x)

        y = y.squeeze().round()
        y_hat = y_hat.squeeze().round()
        print("Y: {} -- Y_hat: {}".format(y, y_hat))

        if torch.squeeze(y) == 1.0:
            positive_examples_count += 1
        else:
            negative_examples_count += 1

        if y == y_hat:
            if y == 1.0:
                true_positive += 1
            else:
                true_negative += 1
        else:
            if y_hat == 1.0:
                false_positive += 1
            else:
                false_negative += 1

        print("--------------------------------------------------------------------------")

    try:
        precision = true_positive / (true_positive + false_positive)
    except ZeroDivisionError:
        precision = 0.0

    try:
        recall = true_positive / (true_positive + false_negative)
    except ZeroDivisionError:
        recall = 0.0

    try:
        f_score = 2 * (precision * recall) / (precision + recall)
    except ZeroDivisionError:
        f_score = 0.0

    f.write("\nTotal of examples: {} (1s: {} 0s: {})".format(len(test_loader), positive_examples_count, negative_examples_count))
    f.write("\nTP: {} - TN: {} - FP: {} - FN: {}".format(true_positive, true_negative, false_positive, false_negative))
    f.write("\nPrecision: {:.4f} Recall: {:.4f} F-Score: {:.4f}".format(precision, recall, f_score))
