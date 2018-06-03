import tensorflow as tf

x = tf.constant(5)
y = tf.constant(6)

result = tf.multiply(x,y)
print(result)
# te see reult we have to run s session in tensorflow
with tf.Session() as sess:
	output =(sess.run(result))
	print(output)
