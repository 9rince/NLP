import tensorflow as tf
import pandas as pd
from textblob import TextBlob
import numpy as np
from tqdm import tqdm
from prep import prep_data
from gru_modules import gru_models 

class DMN_QA():
    """
    the word vectors have size 100
    the max no of words in a passage is 200
    """
    
    def __init__(self):
        self.MAX_P_WORDS = 200       ## max no of words in a passage
        self.WORD_VEC_LEN = 100      ## word vector length
        self.MAX_Q_WORDS = 10        # max no of words in question
        self.MAX_NO_CONCEPTS = 5      # max no of concepts
        self.learning_rate = .1
        
        # variables and placeholders
        self.Passage = tf.placeholder(shape=(self.MAX_P_WORDS,self.WORD_VEC_LEN,1),dtype = tf.float64)
        self.Question = tf.placeholder(shape=(self.MAX_Q_WORDS,self.WORD_VEC_LEN,1),dtype = tf.float64) #sequence for question
        self.EOS_Tag = np.zeros([self.MAX_NO_CONCEPTS],dtype=int)
        # weigts for scalar kernel and memory vector
        self.W_b = tf.get_variable("W_b",shape=(100,100),dtype=tf.float64)
        self.W_1 = tf.get_variable("W_1",shape=(200,702),dtype=tf.float64)
        self.b_1 = tf.get_variable("b_1",shape=(200,1),dtype=tf.float64)
        self.W_2 = tf.get_variable("W_2",shape=(1,200),dtype=tf.float64)
        self.b_2 = tf.get_variable("b_2",shape=(1,1),dtype=tf.float64)
        
        # initial memory state
#         self.memory = tf.placeholder(shape=(100,1),dtype=tf.float64)
        
        #training parameters
        self.no_epoch = 10
        self.df_contxt = pd.read_pickle('./input/squad_contxt.pkl') # context
        self.df_qas = pd.read_pickle('./input/squad_qas.pkl')       # question and answers
        # self.EOS_Tag = [tf.placeholder(tf.int32) for i in range(self.MAX_NO_CONCEPTS)]
        self.snt_wt = tf.placeholder(shape=(self.MAX_NO_CONCEPTS),dtype=tf.float64)
        
    def scalar_gate_value(self,concept):  # debugged and perfect
        z_vector = tf.concat((concept,self.memory),axis=0)
        z_vector = tf.concat((z_vector,self.question),axis=0)
        z_vector = tf.concat((z_vector,tf.multiply(concept,self.question)),axis=0)
        z_vector = tf.concat((z_vector,tf.multiply(concept,self.memory)),axis=0)
        z_vector = tf.concat((z_vector,concept-self.question),axis=0)
        z_vector = tf.concat((z_vector,concept-self.memory),axis=0)
        z_vector = tf.concat((z_vector,tf.matmul(tf.transpose(concept),tf.matmul(self.W_b,self.question))),axis=0)
        z_vector = tf.concat((z_vector,tf.matmul(tf.transpose(concept),tf.matmul(self.W_b,self.memory))),axis=0)
        g_1 =  tf.nn.tanh(tf.add(tf.matmul(self.W_1,z_vector),self.b_1))
        g_scalar =  tf.nn.sigmoid(tf.add(tf.matmul(self.W_2,g_1),self.b_2))
        return g_scalar
        
    def epi_mem_mod(self,concepts,no_of_iterations,mode,mem_states=[]): #debugged
        self.scalar_vector = []
        for i in concepts:
            self.scalar_vector.append(self.scalar_gate_value(i))
        
        concepts_out = mode.mod_gru_unit(h_prev = self.memory,steps=len(concepts),scalar_values=self.scalar_vector,
                          out=[])
        self.memory = concepts_out[-1]
        mem_states.append(concepts_out[-1])

        if no_of_iterations != 0:
            self.epi_mem_mod(concepts,no_of_iterations-1,
                        mode,mem_states)

        return mem_states
    
    def get_sentence_weight(self,q_no):
        y = self.df_contxt['context'][self.df_qas['context_no'][q_no]]
        c = -1
        count = 0
        snt_wt = np.zeros(self.MAX_NO_CONCEPTS)
        if self.df_qas['is_impossible'][q_no]!= True:
            print()
            for i in TextBlob(y).sentences:
                c += (len(i)+1)
                if self.df_qas['Answer_start'][q_no] < c:
                    snt_wt[count] = 1.
                    break
        return snt_wt
    
    def build(self,mode):
        
        # question module 
        Q_module = gru_models(time_steps=self.MAX_Q_WORDS,
                              init_state=np.zeros([self.WORD_VEC_LEN,1]),
                              input_size=100,output_size=100,
                              input_seq=self.Question,tag=1)
        Q_out = Q_module.gru_unit(h_prev=np.zeros([self.WORD_VEC_LEN,1]),
                                  steps=self.MAX_Q_WORDS,out=[])
        
        self.question = Q_out[-1] # final representation of question
        #passage module
        P_module = gru_models(time_steps=self.MAX_P_WORDS,
                              init_state=np.zeros([self.WORD_VEC_LEN,1]),
                              input_size=100,output_size=100,
                              input_seq=self.Passage,tag=2)
        P_out = P_module.gru_unit(h_prev=np.zeros([self.WORD_VEC_LEN,1]),steps=self.MAX_P_WORDS,out=[])
        #episodic memory module
        if mode == 'Sentences':
            concepts = []  # selected E-O-S tags from P_out
            for i in self.EOS_Tag:
                concepts.append(P_out[i])
        elif mode == 'Words':
            concepts = P_out

        self.memory = self.question
        epi_mod = gru_models(time_steps=2,
                             init_state=np.zeros([self.WORD_VEC_LEN,1]),
                             input_size=100,output_size=100,
                             input_seq=concepts,tag=3)
        self.mem_states = self.epi_mem_mod(concepts,10,epi_mod,[])
        
        self.output = self.scalar_vector
        
        return self.output
    
    def train_model(self):
        pred = self.build(mode = 'Sentences')
        cost = self.snt_wt - pred
        init_op = tf.global_variables_initializer()
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cost)
        with tf.Session() as sess:
            sess.run(init_op)
            my_data = prep_data()
            self.MAX_P_WORDS = my_data.max_p_len
            self.MAX_Q_WORDS = my_data.max_q_len
            for i in tqdm(range(self.no_epoch)):
                c = 0
                while c<1000:
                    q_no  = np.random.randint(0,10000)
                    p2v,q2v,eos_tag,ans_snt = my_data.get_vectors(q_no)
                    self.EOS_Tag = eos_tag
                    sess.run(optimizer,feed_dict={self.Passage: p2v ,
                                                 self.Question: q2v ,
                                                 self.snt_wt:ans_snt})

    # def train_model_eager():
    #     tf.enable_eager_execution()

                    
                    
            
            
        
    
        """
        #answer module
        A_module = gru_models(time_steps=len(self.mem_states),
                              init_state=None,
                              question=self.question,
                              input_size=200,output_size=100,
                              input_seq=self.mem_states,tag=4)
        A_out = A_module.answer_gru_unit(h_prev=self.mem_states[-1],
                                         word_prev=np.zeros([self.WORD_VEC_LEN,1]),
                                         out=[]) 
        """
