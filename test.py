import tensorflow as tf
import numpy as np

rnn = tf.contrib.cudnn_rnn.CudnnGRU(1, 2)#, direction='bidirectional')

x = tf.placeholder(shape=(10,1,2),dtype=tf.float32)
out,hid = rnn(x)
print(rnn(x))
hid= hid[0]
# z = tf.reshape(hid,[10,2,1])
sess = tf.Session()
sess.run(tf.global_variables_initializer())
a,b = sess.run([out,hid],feed_dict={x:np.random.random(size = (10,1,2))})
# sess.close()
# print(c)
# print(b.reshape((10,2,1)))
print(b)
# print(y[1])