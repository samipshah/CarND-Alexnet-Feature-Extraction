import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet
from sklearn.utils import shuffle
from keras.datasets import cifar10

nb_classes = 10
rate = 0.001
EPOCHS = 1
BATCH_SIZE = 128

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# y_train.shape is 2d, (50000, 1). While Keras is smart enough to handle this
# it's a good idea to flatten the array.
y_train = y_train.reshape(-1)
y_test = y_test.reshape(-1)

# TODO: Load traffic signs data.
#with open('./train.p', 'rb') as f:
#    traindata = pickle.load(f)

from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,
                                                      test_size=0.3,
                                                      random_state=42, stratify
                                                      = y_train)
# TODO: Split data into training and validation sets.
#X_train, X_test, y_train, y_test = train_test_split( traindata["features"], 
#                                                    traindata["labels"], 
#                                                    test_size=0.33, 
#                                                    random_state=42)

# TODO: Define placeholders and resize operation.
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
resized = tf.image.resize_images(x,(227,227))

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
shape = (fc7.get_shape().as_list()[-1], nb_classes)  # use this shape for the weight matrix
fc8W = tf.Variable(tf.random_normal(shape, stddev=0.1), tf.float32)
fc8b = tf.Variable(tf.zeros(shape[1]), tf.float32)
logits = tf.matmul(fc7, fc8W) + fc8b

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
onehot_y=tf.one_hot(y, nb_classes)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=onehot_y,logits=logits)
# Loss Function reduce mean (reduce mean over the batch)
loss_function = tf.reduce_mean(cross_entropy)
# Adam Optimizer (optimizer for reduce mean function)
optimizer=tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_function)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(onehot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


# TODO: Train and evaluate the feature extraction model.
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    validation_accuracy = 0.0
    for i in range(EPOCHS):
        # early stop for learning
        X_train_shuffled, y_train_shuffled = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train_shuffled[offset:end], y_train_shuffled[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
        saver.save(sess, './cifer10')
        print("Model saved")
        training_accuracy = evaluate(X_train, y_train)
        validation_accuracy = evaluate(X_test, y_test)
        print("EPOCH {} ...".format(i+1))
        print("Training Accuracy = {:.3f}".format(training_accuracy))
        print("Test Accuracy = {:.3f}".format(validation_accuracy))
        print()

