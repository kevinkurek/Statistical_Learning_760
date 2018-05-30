# TensorFlow

# TensorFlow is a computation graph composed of a set of nodes. Each
# node represents an operation that may have zero or more input or output.
# The values that flow through the edges of the computation graph are called
# tensors (hence the name).

# Implement basic TensorFlow z = w*x + b
import tensorflow as tf

# Create a Graph
g = tf.Graph()

with g.as_default():
  x = tf.placeholder(dtype=tf.float32, shape=(None), name='x')
  w = tf.Variable(2.0, name='weight')
  b = tf.Variable(0.7, name='bias')

  z = w * x + b

  init = tf.global_variables_initializer()

# Create a session and pass in graph g
with tf.Session(graph=g) as sess:
  # Initialize w and b:
  sess.run(init)
  # Evaluate z:
  for t in [1.0, 0.6, -1.8]:
    print('x =%4.1f --> z =%4.1f'%(t, sess.run(z, feed_dict={x:t})))
