import keras
import tensorflow as tf
import keras.backend as K
# a = tf.ones([10], dtype=tf.int8)
#
# b = tf.constant(1)
# c = tf.zeros((1))
# d = K.zeros([1])
# print(d)
#
# with tf.Session() as sess:
#     print(sess.run(a))
#     print(sess.run(a-1))
#     print(sess.run(b))
#     print(sess.run(c))

# a = {'a1': 1, 'a2': 2, 'a3': 3}
# b = {'b1': 11, 'b2': 12, 'b3': 13}
# print(dict([('a', '1')]))
# print(list(a.items()) + list(b.items()))
# print(dict(list(a.items()) + list(b.items())))
#
# c = [1]
# d = [2]
# print(c + d)

a = K.ones(shape=[128, 2, 512, 64])
b = K.ones(shape=[128, 2, 512, 64])
c = K.batch_dot(a, b, axes=[3, 3])
d = K.batch_dot(a, b, axes=[2, 2])
b = K.ones(shape=[128, 2, 64, 512])
e = K.dot(a, b)
# print(K.int_shape(e))

a1 = K.ones(shape=[128, 512, 768])
b1 = K.ones(shape=[21128, 768])
print(K.int_shape(b1))
b1 = K.transpose(b1)
print(K.int_shape(b1))
c1 = K.dot(a1, b1)
print(K.int_shape(c1))

