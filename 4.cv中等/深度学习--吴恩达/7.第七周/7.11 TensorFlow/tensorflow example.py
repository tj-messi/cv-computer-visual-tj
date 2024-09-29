import numpy as np
import tensorflow as tf

w=tf.Variable([2,3],stddev=1,seed=1)
b=tf.Variable(tf.zeros([3]))
x=tf.constant([[0.7,0.9]])
y=tf.nn.softmax(tf.matmul(x,w)+b)

init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(y))
