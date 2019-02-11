import tensorflow as tf 

class gru_models():
    """
        all gru models required for the model goes here!
    """
    def __init__(self,time_steps,init_state,input_size,tag,
                 output_size,question=None,final_mem=None,input_seq=None):
        
        # inputs required for sequential memory vectors e_i 
        self.time_steps = time_steps              # equal to max no concepts
        self.init_state = init_state              # initial secondary input ideally zero tensor
        self.input_seq = input_seq                # input seq to be evaluated
        self.input_size = input_size              # size of input vector 
        self.output_size = output_size            # size of output required
        
        # inputs required for Answer modules
        self.question = question
        self.final_mem = final_mem
        self.word_weight = tf.get_variable("word_w_{}".format(tag),[output_size,output_size],dtype=tf.float64)
        self.word_bias = tf.get_variable("word_b_{}".format(tag),[output_size,1],dtype=tf.float64)
        
        # parameters of gru
        self.W_z,self.W_r,self.W_h = tf.get_variable("update_w_{}".format(tag),[output_size,input_size],dtype=tf.float64),tf.get_variable("reset_w_{}".format(tag),[output_size,input_size],dtype=tf.float64),tf.get_variable("out_w_{}".format(tag),[output_size,input_size],dtype=tf.float64)
        self.U_z,self.U_r,self.U_h = tf.get_variable("update_u_{}".format(tag),[output_size,output_size],dtype=tf.float64),tf.get_variable("reset_u_{}".format(tag),[output_size,output_size],dtype=tf.float64),tf.get_variable("out_u_{}".format(tag),[output_size,output_size],dtype=tf.float64)
        self.b_z,self.b_r,self.b_h = tf.get_variable("update_b_{}".format(tag),[output_size,1],dtype=tf.float64),tf.get_variable("reset_b_{}".format(tag),[output_size,1],dtype=tf.float64),tf.get_variable("out_b_{}".format(tag),[output_size,1],dtype=tf.float64)
    
    """ Normal GRU unit """
    def gru_unit(self,h_prev,steps,out=[]):
        index = steps - self.time_steps 
        input_vector = self.input_seq[index]
        # operations 
        z_t = tf.nn.sigmoid(tf.matmul(self.W_z,input_vector)+tf.matmul(self.U_z,h_prev)+self.b_z)
        r_t = tf.nn.sigmoid(tf.matmul(self.W_r,input_vector)+tf.matmul(self.U_r,h_prev)+self.b_r)
        h_t = tf.multiply(z_t,h_prev)+tf.multiply(1.-z_t,tf.nn.sigmoid(tf.matmul(self.W_h,input_vector)+tf.matmul(self.U_h,tf.multiply(r_t,h_prev))+self.b_h))
        out.append(h_t)
        self.time_steps -= 1
        if self.time_steps == 0:
            return(out)
        else:
            return self.gru_unit(h_prev = h_t,out = out,steps = steps) 
    
    
    """
    it's a modified GRU !!!
    """
    def mod_gru_unit(self,h_prev,steps,scalar_values,out=[],index=0):
        input_vector = self.input_seq[index]
        # operations
        z_t = tf.nn.sigmoid(tf.matmul(self.W_z,input_vector)+tf.matmul(self.U_z,h_prev)+self.b_z)
        r_t = tf.nn.sigmoid(tf.matmul(self.W_r,input_vector)+tf.matmul(self.U_r,h_prev)+self.b_r)
        h_t = tf.multiply(z_t,h_prev)+tf.multiply(1.-z_t,tf.nn.sigmoid(tf.matmul(self.W_h,input_vector)+tf.matmul(self.U_h,tf.multiply(r_t,h_prev))+self.b_h))
        h_t = (scalar_values[index]*h_t) + ((1-scalar_values[index])*h_prev)
        out.append(h_t)
        index += 1
        if steps-index == 0:
            return(out)
        else:
            return self.mod_gru_unit(h_prev = h_t,out = out,steps = steps,
                                     scalar_values=scalar_values,index=index) 
        
    """ It's a hybrid GRU for answer module """
    def answer_gru_unit(self,h_prev,word_prev,out=[]):
        input_vector = tf.concat([word_prev,self.question],axis=0)
        z_t = tf.nn.sigmoid(tf.matmul(self.W_z,input_vector)+tf.matmul(self.U_z,h_prev)+self.b_z)
        r_t = tf.nn.sigmoid(tf.matmul(self.W_r,input_vector)+tf.matmul(self.U_r,h_prev)+self.b_r)
        h_t = tf.multiply(z_t,h_prev)+tf.multiply(1.-z_t,tf.nn.sigmoid(tf.matmul(self.W_h,input_vector)+tf.matmul(self.U_h,tf.multiply(r_t,h_prev))+self.b_h))
        word_pred = tf.nn.softmax(tf.matmul(self.word_weight,h_t)+self.word_bias)
        out.append(word_pred)
        self.time_steps -= 1
        if self.time_steps == 0:
            return(out)
        else:
            return self.answer_gru_unit(h_prev = h_t,word_prev = word_pred,out = out) 