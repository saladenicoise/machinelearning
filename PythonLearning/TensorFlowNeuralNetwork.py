import tensorflow as tf
import numpy as np

#Input Data (100 Phony data points)
x_data = np.float32(np.random.rand(2,100))
y_data = np.dot([0.1, 0.2], x_data) + 0.3

#All Input data is considered a tensor, a tensor is a 3 dimensional matrix

#Constructing a linear model
b = tf.Variable(tf.zeros(1)) #Bias
s = tf.Variable(tf.random_uniform([1, 2], -1, 1))
y = tf.matmul(s, x_data) + b # mx + b

#Performing Gradient Descent, actual machine learning part. A valley with a ball i it and we want the valley to be so that the ball always fall in the midpoint, smoothing 3 dimensional path that the tensors are following
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

#In tensorflow we wrap the data in a graph, we initialize the variables here:
init = tf.initialize_all_variables()

#Launch the graph
sess = tf.Session() # A session is an event where we compute
sess.run(init)

#Fitting/Training Process
for step in range(0, 200):
    sess.run(train)
    if step % 20 == 0:
        print(step, see.run(s), sess.run(b))
