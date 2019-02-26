# import tensorflow as tf
# from gensim.models import Word2Vec
import pandas as pd
from textblob import TextBlob
from gensim.models import Word2Vec
import numpy as np
# from tqdm import tqdm

class prep_data:
  def __init__(self,mode):
    
    if mode=='training':
      self.contxt = pd.read_pickle('./current_input/squad_contxt.pkl')
      self.qas = pd.read_pickle('./current_input/squad_qas.pkl')
    if mode=='testing':
      self.contxt = pd.read_pickle('./current_input/squad_contxt.pkl') 
      self.qas = pd.read_pickle('./current_input/squad_qas.pkl')

    self.qas['max'] = self.qas['word_len']+self.qas['snt_len']
    self.max_q_len = max(self.qas['max'])
    self.contxt['max'] = self.contxt['word_len']+self.contxt['snt_len']
    self.max_p_len = max(self.contxt['max'])
    self.word_vec = Word2Vec.load('model.bin')
    self.max_p_snt_len = max(self.contxt['snt_len'])
    self.max_q_snt_len = max(self.qas['snt_len'])
    self.word_vec_size = 100
    self.q_list = list(self.qas['q_no'])
  
  def get_snt(self,q_no):
    ans,p_no = self.qas.loc[self.qas['q_no'] == q_no]['Answer_start'][q_no],self.qas.loc[self.qas['q_no'] == q_no]['context_no'][q_no]
    para = str(self.contxt.loc[self.contxt['context_no']== p_no]['context'][p_no])
    c = -1
    snt = TextBlob(para).sentences
    p_len = len(snt)
    for j in range(p_len):
      c += len(snt[j])+1
      if ans<c:
        df_qas.loc[df_qas['q_no'] == i]['ans_snt'][i] = j
        return j

  def get_vectors_1(self,q_no):   # this function converts context 2 vector with <EOS> tag
    tb_p = TextBlob(self.contxt['context'][self.qas['context_no'][q_no]])#TextBlob(passage)
    tb_q = TextBlob(self.qas['Question'][q_no])
    tb_a = TextBlob(self.qas['Answer_text'][q_no]).words
    p2v = []
    q2v = []
    a2v = []
    snt_ans = self.get_snt(q_no)
    eos_tag = []
    c = 0
    for i in tb_p.sentences:
      words = TextBlob(str(i)+" E-O-S").words
      x_flag = 0
      x_old = 0
      for j in words:
          p2v.append(self.word_e[j])

      c += 1
    
    for k in (tb_q+" E-O-S").words:
      q2v.append(self.word_e[k])
    
    p2v = self.pad_it(p2v,self.max_p_len)
    q2v = self.pad_it(q2v,self.max_q_len)
    ans_snt = np.zeros([self.max_p_snt_len]) 
    ans_snt[snt_ans] = 1.
        
    return np.array(p2v),np.array(q2v),eos_tag,ans_snt
  def get_vectors(self):
    q_no = np.random.randint(len(self.q_list))
    try:
      passage = self.contxt['context'][self.qas['context_no'][q_no]]
      quest = self.qas['Question'][q_no]#).words
      start = self.qas['Answer_start'][q_no]
      ans = self.qas['Answer_text'][q_no]#).words
      beg = passage[:start]#).words
      end = passage[start+len(ans):]#).words 
      # print('--'*100)
      # print(passage)
      # print('--'*100)
      # print(beg)
      # print('--'*100)
      # print(ans)
      # print('--'*100)
      # print(end)
      # print('--'*100)
      # print(TextBlob(passage).words)
      # print('--'*100)
      # print(TextBlob(beg).words)
      # print('--'*100)
      # print(TextBlob(ans).words)
      # print('--'*100)
      # print(TextBlob(end).words)
      # print('--'*100)

      pas = []
      pas_wt = []
      qs = []
      for i in list(TextBlob(beg).words):
        pas.append(self.word_vec[i].reshape(100,1))
        pas_wt.append(0)
      for j in list(TextBlob(ans).words):
        pas.append(self.word_vec[j].reshape(100,1))
        pas_wt.append(1)
      for k in list(TextBlob(end).words):
        pas.append(self.word_vec[k].reshape(100,1))
        pas_wt.append(0)

      pas,pas_wt = self.pad_it(pas,pas_wt)

      for l in list(TextBlob(quest).words):
        qs.append(self.word_vec[l].reshape(100,1))

      while len(qs)<30:
        qs.append(np.zeros([100,1]))

      return pas,pas_wt,qs
    except KeyError:
      return self.get_vectors()

  def pad_it(self,pas,pas_wt,max_len = 50):

    while len(pas)<max_len:
      pas.append(np.zeros([100,1]))
      pas_wt.append(0)
      # print(len(pas))

    len(pas)

    return pas,pas_wt