
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data
import pictureslicernew
import tensorflow as tf
import numpy as np
import cut
import time

FLAGS = None
restore=False
newData=False
    
def main(_):
  # Import data
  numofpixels=200
  while(numofpixels%4!=0):  #unfortunaly we can't cut pixels in half MUST BE divisble by 4 for my pictureslicer to work
     numofpixels+=1
  #temp=cut.cutter("donaldtrump.jpg", numofpixels)
  #temp.sliceup()
  segmentsForTesting=None
  segmentsForTraining=None
  segmentsForTraining=pictureslicernew.newsegs(numofpixels, "training", True) #this is the size may or may not be a good starting point
  segmentsForTesting=pictureslicernew.newsegs(numofpixels, "test", True)
  if newData:
   segmentsForTraining.calculatesegments(100,"precalctraining")
   segmentsForTesting.calculatesegments(100,"precalctesting")

  matrixsize=((numofpixels/2)*(numofpixels/2))*3
  x,W,b,y, y_, saver, cross_entropy, train_step, sess=None, None, None, None, None, None, None, None, None
  # Create the model
  if  not restore:
   x = tf.placeholder(tf.float32, [None, matrixsize], name="x")
   weights=np.random.uniform((matrixsize, 2)   
   W = tf.Variable(weights, name="W")
   b = tf.Variable(tf.zeros([2]), name="b")
   y = tf.matmul(x, W) + b

  # Define loss and optimizer
   y_ = tf.placeholder(tf.float32, [None, 2], name="y")

   saver = tf.train.Saver()
   cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y), name="ent")
   train_step = tf.train.GradientDescentOptimizer(0.00001).minimize(cross_entropy)

   sess = tf.InteractiveSession()

   tf.global_variables_initializer().run()

  else:
    sess = tf.InteractiveSession()
    saver = tf.train.import_meta_graph('model.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./'))
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    W = graph.get_tensor_by_name("W:0")
    print(sess.run('W:0'))
    b = graph.get_tensor_by_name("b:0")
    y = tf.matmul(x, W) + b
    y_ = graph.get_tensor_by_name("y:0")
    saver = tf.train.Saver()
    cross_entropy = graph.get_tensor_by_name("ent:0")
    train_step = tf.train.GradientDescentOptimizer(0.00001).minimize(cross_entropy)
    #tf.global_variables_initializer().run() 
  epocs=0
  segmentsForTraining.gatherFiles("precalctraining")
  segmentsForTesting.gatherFiles("precalctesting")
  trainingDeepCopy=segmentsForTraining.files[:]
  testingDeepCopy=segmentsForTesting.files[:]
  while epocs<1000:
    segmentsForTraining.files=trainingDeepCopy[:]
    segmentsForTesting.files=testingDeepCopy[:]
    epocs+=1
    print ("TESTING FOR ",epocs)
    while len(segmentsForTraining.files)>0:
     #print ("I TEST ARROUND") 
     arrayofconnections=segmentsForTraining.getBatch()
     for tup in arrayofconnections :
      batch_xs = [np.asarray([tup[0]]).flatten()]
      batch_ys = np.asarray([tup[1]])
      sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    print ("TRAINING FOR ",epocs)
    truescore=0.0
    rounds=0.0
    while len(segmentsForTesting.files)>0:
     arrayofconnections=segmentsForTesting.getBatch()
     images=[np.asarray(i[0]).flatten() for i in arrayofconnections]# flattern for now will need to convert it to 2d. Just a starting point
     tags=[i[1] for i in arrayofconnections]
    # Test trained model
     tags=np.asarray(tags)
     images=np.asarray(images)
     print( "I feed it tags of shape ", tags.shape, " and I feed it images of shape ", images.shape)
     correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
     score=  sess.run(accuracy, feed_dict={x: images,
                                      y_: tags})
     truescore+=score
     rounds+=1.0
    print("My accuracy is ", truescore/rounds," for epoch ", epocs )
  save_path=saver.save(sess, "model.ckpt")

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  parser.add_argument('--restore', type=bool, default=True, help="restore the model")
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
