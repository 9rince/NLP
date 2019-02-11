# import tensorflow as tf
# from gensim.models import Word2Vec
import pandas as pd
from textblob import TextBlob
from gensim.models import Word2Vec
import numpy as np
# from tqdm import tqdm

class prep_data():
    """
    This class gives 
    """
    def __init__(self):
        
        self.contxt = pd.read_pickle('./current_input/squad_contxt_train.pkl')
        self.qas = pd.read_pickle('./current_input/squad_qas_train.pkl')
        self.qas['max'] = self.qas['word_len']+self.qas['snt_len']
        self.max_q_len = max(self.qas['max'])
        self.contxt['max'] = self.contxt['word_len']+self.contxt['snt_len']
        self.max_p_len = max(self.contxt['max'])
        self.word_e = Word2Vec.load('model.bin')
        self.max_p_snt_len = max(self.contxt['snt_len'])
        self.max_q_snt_len = max(self.qas['snt_len'])
        self.word_vec_size = 100
    
    def get_snt(self,q_no):
        ans,p_no = self.qas.loc[self.qas['q_no'] == q_no]['Answer_start'][q_no],self.qas.loc[self.qas['q_no'] == q_no]['context_no'][q_no]
        para = str(self.contxt.loc[self.contxt['context_no']== p_no]['context'][p_no])
        c = -1
        snt = TextBlob(para).sentences
        p_len = len(snt)
        for j in range(p_len):
	        c += len(snt[j])+1
	        if ans<c:
	#             df_qas.loc[df_qas['q_no'] == i]['ans_snt'][i] = j
	            return j

    def get_vectors(self,q_no):   # this function converts context 2 vector with <EOS> tag
        tb_p = TextBlob(self.contxt['context'][self.qas['context_no'][q_no]])#TextBlob(passage)
        tb_q = TextBlob(self.qas['Question'][q_no])
        p2v = []
        q2v = []
        eos_tag = []
        c = 0
        for i in tb_p.sentences:
            words = TextBlob(str(i)+" E-O-S").words
            for j in words:
                p2v.append(self.word_e[j])
                if j=='E-O-S':
                    eos_tag.append(c)
            c += 1
        for k in (tb_q+" E-O-S").words:
            q2v.append(self.word_e[k])
        
        p2v = self.pad_it(p2v,self.max_p_len)
        q2v = self.pad_it(q2v,self.max_q_len)
        ans_snt = np.zeros([self.max_p_snt_len]) 
        ans_snt[self.get_snt(q_no)] = 1.
            
        return np.array(p2v),np.array(q2v),eos_tag,ans_snt

    def pad_it(self,my_list,max_len):

    	while len(my_list)<max_len:
    		my_list.append(np.zeros([self.word_vec_size]))

    	return my_list