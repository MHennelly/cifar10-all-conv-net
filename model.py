import tensorflow as tf

def createPlaceholder(shape):
    return tf.placeholder(dtype=tf.float32, shape=shape)

def createWeight_Bias(shape):
    w = tf.Variable(tf.truncated_normal(shape, stddev = 0.3))
    b = tf.Variable(tf.truncated_normal([shape[-1]], stddev = 0.3))
    return w,b

def createConvLayer(input, shape, stride):
    weights, bias = createWeight_Bias(shape)
    conv = tf.nn.conv2d(
        input=input,
        filter=weights,
        strides=[1,stride,stride,1],
        padding="SAME"
    )
    conv += bias
    mean, variance = tf.nn.moments(conv, axes=[0,1,2,3])
    conv = tf.nn.batch_normalization(conv, mean, variance, None, None, 0.001)
    conv = tf.nn.relu(conv)
    conv = tf.nn.dropout(conv,keep_prob)
    return conv

def createDenseLayer(input, wShape):
    weights, bias = createWeight_Bias(wShape)
    return tf.matmul(input,weights)+bias

keep_prob = tf.placeholder(tf.float32)
IL = tf.placeholder(dtype=tf.float32, shape=(None,32,32,3))

CL1 = createConvLayer(IL, (3,3,3,96),1)

CL2 = createConvLayer(CL1, (3,3,96,96),1)

CL3 = createConvLayer(CL2,(3,3,96,96),2)

CL4 = createConvLayer(CL3,(3,3,96,192),1)

CL5 = createConvLayer(CL4,(3,3,192,192),1)

CL6 = createConvLayer(CL5,(3,3,192,192),2)

CL7 = createConvLayer(CL6,(3,3,192,192),1)

CL8 = createConvLayer(CL7,(1,1,192,192),1)

CL9 = createConvLayer(CL8,(1,1,192,10),1)

logits = tf.reshape(tf.nn.avg_pool(CL9, (1,8,8,1), (1,1,1,1), "VALID"),[-1,10])

OL = tf.nn.softmax(logits)

OL_ = tf.placeholder(dtype=tf.float32, shape=(None,10))

learning_rate = 0.01
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    logits = logits,
    labels = OL_
))
optimizer = tf.train.AdamOptimizer(
    learning_rate = learning_rate,
).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(OL,1), tf.argmax(OL_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
