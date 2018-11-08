#!/usr/bin/python3
import os, random, sys, time
from datetime import datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from setup_data import get_data

# Create directory to save image predictions by uncommenting:
# os.mkdir('img_prediction_dir')

# Define system parameters and
# save logs for each run in separate directory

beginTime = time.time()

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('train_dir', 'tf_logs', 'Directory to put the training data.')

logdir = FLAGS.train_dir + '/' + datetime.now().strftime('%Y%m%d-%H%M%S') + '/'

# cmd line: 
# $ tensorboard --logdir ./  
# to view tensorboard output


# Prepare data
data_sets = get_data()

images_train = data_sets['images_train']
labels_train = data_sets['labels_train']

images_val = data_sets['images_val']
labels_val = data_sets['labels_val']

images_test = data_sets['images_test']


# Model Parameters 
def ceil(a,b):
    return -(-a//b)

N = len(images_train)

iterations = 1000

batch_size = ceil(N, ceil(N, 100))

IMG_SIZE = 300*300

learning_rate = 0.001


# Define Model Inputs
x = tf.placeholder(tf.float32, shape=[None, IMG_SIZE])
y_ = tf.placeholder(tf.int64, [None,2])

W = tf.get_variable("weights", shape=[IMG_SIZE, 2])
b = tf.get_variable("bias", shape=[2], initializer=tf.constant_initializer(0.01))

# Model
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Training Operations
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=y_))
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# Training Metrics
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# Summary data for Tensorboard
tf.summary.scalar('accuracy', accuracy)
merged = tf.summary.merge_all()
saver = tf.train.Saver()

# ======================== Initialize TF Session =================================
# Launch the graph
sess = tf.Session()

with sess.as_default():
    tf.global_variables_initializer().run()
    summary_writer = tf.summary.FileWriter(logdir, sess.graph)
    
    for step in range(iterations):
        indices = np.random.choice(images_train.shape[0], batch_size)
        xs = images_train[indices].reshape([-1, IMG_SIZE])
        ys = labels_train[indices].reshape([-1,2])
        
        if step % 100 == 0:
            train_acc = sess.run(accuracy, feed_dict={x: xs, y_: ys})
            print('Step {:5d}: training accuracy {:g}'.format(step, train_acc))
            
            summary, acc = sess.run([merged, accuracy], feed_dict={x: xs, y_: ys})
            summary_writer.add_summary(summary, step)
        
        sess.run([train_step, loss], feed_dict={x: xs, y_: ys})

        # Periodically save checkpoint
        if (step + 1) % 1000 == 0:
            checkpoint_file = os.path.join(FLAGS.train_dir, 'checkpoint')
            saver.save(sess, checkpoint_file, global_step=step)
            print('Saved checkpoint')          
    
    print("\nTraining complete!")

    xs_val = np.reshape(images_val, [-1, IMG_SIZE])
    ys_val = np.reshape(labels_val, [-1,2])
    
    validation_acc = sess.run(accuracy, feed_dict={x: xs_val, y_: ys_val})
    
    print('Validation accuracy {:g}'.format(validation_acc))

    endTime = time.time()
    print('Total time: {:5.2f}s'.format(endTime - beginTime))

#========================== Prediction ===========================
# Save randomly selected images from test data with predicted class 
# in filename
     
    num = [random.randint(0, images_test.shape[0]) for i in range(100)]
    test_img = images_test[num].reshape([-1, IMG_SIZE])      

    category = tf.argmax(y, 1) # choose by maximum score
    classification = category.eval({x: test_img})


    print("scores:", y.eval({x: test_img}))
    print('class:', classification)


    for i, img_i in enumerate(test_img):
        plt.xlabel('Test prediction: {classification[i]}')
        plt.imsave(os.path.join('img_prediction_dir', 'img-{}-class-{}.png'.format(str(i),\
         classification[i])), img_i.reshape(300, 300), cmap=plt.cm.binary)


sess.close()




