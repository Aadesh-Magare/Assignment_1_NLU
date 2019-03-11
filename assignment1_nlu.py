import nltk
from nltk.corpus import reuters
import numpy as np
from keras.preprocessing.text import Tokenizer,one_hot
from keras.preprocessing import sequence
from keras.utils import np_utils
import tensorflow as tf
from collections import Counter
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random, sys, pickle
from numpy import dot
from numpy.linalg import norm
from scipy import stats
import matplotlib.pyplot as plt

def draw_graph(be, filename):
  x = range(0, 25)
  be['x'] = x
  y1, y2, y3 = be.values()
  be['y1'] = y1
  be['y2'] = y2
  be['y3'] = y3

  plt.plot( 'x', 'y1', data=be)
  plt.plot( 'x', 'y2', data=be)
  plt.plot( 'x', 'y3', data=be)
  # plt.legend(['x', 'B:128', 'B:256', 'B:512'], loc='upper left')
  plt.savefig(filename)
  plt.show()
  
def find_cosine(v1, v2):
  return np.abs(dot(v1, v2) / (norm(v1) * norm(v2)))

def check_simlex(path, sim_file):
  word_embd = {}
  with open(path, 'r') as embd_file:
    j = 0
    for line in embd_file.readlines():
      ws = line.split()
      w = ws[0]
      # print(j)
      j += 1
      emb = np.array(ws[1:], dtype=float)
      word_embd[w] = emb
  print('words read', len(word_embd))
  # print(word_embd.items())
  sim_scores = []
  embd_scores = []
  with open(sim_file, 'r') as sim_file:
    for line in sim_file.readlines():
      w1, w2, score = line.split()
      try:
        w1 = word_embd[w1]
        w2 = word_embd[w2] 
        sim_scores.append(float(score))
        embd_scores.append(find_cosine(w1, w2))
      except KeyError as key:
        pass
        # print('error', key)

  # print(len(sim_scores), len(embd_scores))
  print('Spearman Coeffcient:' ,stats.spearmanr(sim_scores, embd_scores))
  return word_embd

def check_analogy(word_embd, index_word, k, embds):
  filename = 'analogy.txt'
  count = 0
  correct = 0
  W = word_embd.values()
  # embds = (model.center_embd.weight.data.numpy() + model.context_embd.weight.data.numpy()) / 2
   
  with open(filename, 'r') as ana_file:
    for line in ana_file.readlines():
      if line.startswith(':'):
        continue
      ws = list(map(lambda x: x.lower(), line.split()))
      if all(w in word_embd for w in ws):
        w1 = word_embd[ws[0]]
        w2 = word_embd[ws[1]]
        w3 = word_embd[ws[2]]
        w4 = word_embd[ws[3]]

        w_out = w1 - w2 + w3
        closest_word_ids = np.argsort(np.abs([np.dot(w_out, v) for v in embds]))[-k:]
        # closest_word_ids = np.argsort(np.abs(np.dot(w_out, W)))[-k:]
        closest_words = [index_word[id] for id in closest_word_ids if id > 0]
        # print(closest_words)        
        if ws[3] in closest_words:
          # print(correct)
          correct += 1
        count += 1
        # print(count)

  print('Accuracy:', correct / count, 'K similarity: ', k)

def main(run_mode):
  model_file = 'w2v.pyt'
  filename = 'embd.save'
  id_word = 'id_word_dict.save'

  if run_mode == '-test':
    model = torch.load(model_file)
    embds = (model.center_embd.weight.data.numpy() + model.context_embd.weight.data.numpy()) / 2
    word_embd = check_simlex(filename, 'simlex999.txt')
    with open(id_word, 'rb') as handle:
      index_word_dict = pickle.load(handle)
    check_analogy(word_embd, index_word_dict, 20, embds)
  else:
    be = {}
    # for w_size in [2, 4, 6]:
    for batch_size in [128]:
    # for batch_size in [128, 256, 512]:
      batch_size = 256
      embd_dimensions = 300
      w_size = 2
      neg_size = 5
      sentences_word_index, word_index_dict, index_word_dict = pre_process()
      v_size = len(word_index_dict)
      print('Vocab: ', v_size)
      w_c_nc = create_training_data(sentences_word_index, w_size, neg_size, batch_size)
      batches = create_batches(w_c_nc, batch_size)
      model, losses = train(batches, batch_size, v_size, embd_dimensions)
      embds = model.save_embeddings(filename, index_word_dict)
      torch.save(model, model_file)
      with open(id_word, 'wb') as handle:
        pickle.dump(index_word_dict, handle)
      word_embd = check_simlex(filename, 'simlex999.txt')
      check_analogy(word_embd, index_word_dict, 20, embds)
      be[batch_size] = losses

    draw_graph(be, 'graph' + str('w_size'))

def pre_process():
  raw_sentences = reuters.sents(reuters.fileids())
  tokenizer = Tokenizer()
  tokenizer.fit_on_texts(raw_sentences)
  count_thresh = 5
  low_count_words = [w for w,c in tokenizer.word_counts.items() if c < count_thresh]

  for w in low_count_words:
    del tokenizer.word_index[w]
    del tokenizer.word_docs[w]
    del tokenizer.word_counts[w]

  word_index_dict = tokenizer.word_index
  index_word_dict = {word_index_dict[word] : word for word in word_index_dict }
  sentences_word_index = tokenizer.texts_to_sequences(raw_sentences)
  return sentences_word_index, word_index_dict, index_word_dict

def get_negative_samples(neg_samples, no_of_samples):
  idxs = random.sample(neg_samples, no_of_samples)
  return idxs

def create_training_data(sentences_word_index, w_size, neg_size, batch_size):  
  words = []
  contexts = [] 
  neg_contexts = []
  w_c_nc = {}
  for sentence in sentences_word_index:
    l = len(sentence)
    for i in range(l):
      s = i - w_size
      e = i + w_size
      word = sentence[i]
      context = [sentence[j] for j in range(s, e+1) if 0 <= j < l and j != i]
      ctx = set(context)
      neg_samples = [sentence[j] for j in range(0, l) if j < s or j > e]
      if(neg_size <= len(neg_samples)):
        for c in ctx:
          if (word,c) not in w_c_nc:
            neg = get_negative_samples(neg_samples, neg_size)
            w_c_nc[(word,c)] = neg
  
  return w_c_nc

def create_batches(w_c_nc, batch_size):
  batches = []
  ws = []
  cs = []
  ncs = []
  j = 0
  for key, value in w_c_nc.items():
    w, c = key
    nc = value
    if j == batch_size:
      batches.append([ws, cs, ncs])
      ws.clear()
      cs.clear()
      ncs.clear()
      j = 0
    ws.append(w)
    cs.append(c)
    ncs.append(nc)
    j += 1
  
  return batches

class skim_gram(nn.Module):
  
  def __init__(self, vocab_size, embd_dimension):
    super(skim_gram, self).__init__()
    self.vocab_size = vocab_size
    self.embd_dimension = embd_dimension
    self.center_embd = nn.Embedding(vocab_size, embd_dimension, sparse=True)
    self.center_embd.weight.data.uniform_(-1, 1)
    self.context_embd = nn.Embedding(vocab_size, embd_dimension, sparse=True)
    self.context_embd.weight.data.uniform_(-1, 1)
    
  def forward(self, center, positive_context, negative_context, batch_size):
    center_embd = self.center_embd(center)
    positive_ctx_embd = self.context_embd(positive_context)
    negative_ctx_embd = self.context_embd(negative_context)
    
    pos_score = torch.mul(center_embd, positive_ctx_embd)
    pos_score = F.logsigmoid(torch.sum(pos_score, dim=1)).squeeze()
    
    neg_score = torch.bmm(negative_ctx_embd, center_embd.unsqueeze(2)).squeeze()
    neg_score = torch.sum(neg_score, dim=1)
    neg_score = F.logsigmoid(-1 * neg_score).squeeze()
    
    return -1 * (torch.sum(pos_score) + torch.sum(neg_score)) / batch_size

  def save_embeddings(self, path, idx_word):
    embds = (self.center_embd.weight.data.numpy() + self.context_embd.weight.data.numpy()) / 2
    # print(embds)
    with open(path, 'w') as write_file:
      for idx, word in idx_word.items():
          try:
            e = embds[idx]
            e = ' '.join(map(lambda x: str(x), e))
            write_file.write('%s %s\n' % (word, e))
          except IndexError as id:
            print(idx, word)
    print('written to file')
    return embds
    
def train(batches, batch_size, v_size, embd_dimensions):
  learning_rate = 0.01
  epochs = 25
  model = skim_gram(v_size, embd_dimensions)
  model.to(device)
  optimizer = optim.SGD(model.parameters(), lr = learning_rate)
  losses = []
  for epoch in range(epochs):
      total_loss = 0
      for batch in batches:
          word, pos_ctx, neg_ctx  = batch
          word = Variable(torch.LongTensor(word)).to(device)
          pos_ctx = Variable(torch.LongTensor(pos_ctx)).to(device)
          neg_ctx = Variable(torch.LongTensor(neg_ctx)).to(device)
          model.zero_grad()
          loss = model(word, pos_ctx, neg_ctx, batch_size)
          loss.backward()
          optimizer.step()
          total_loss += loss.data.item()
      print("Epoch: ", epoch, "Loss: ", total_loss)
      losses.append(total_loss)

  return model, losses

if __name__ == '__main__':
  random.seed(111)
  # nltk.download('reuters')
  # nltk.download('punkt')
  device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
  print('using device:', device)
  main(sys.argv[1])

