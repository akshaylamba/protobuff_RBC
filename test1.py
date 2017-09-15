from keras.models import Sequential
from keras.layers import Reshape, Activation, Input, MaxPooling2D, BatchNormalization, Flatten, Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import SGD,Adam
import matplotlib.pyplot as plt
import numpy as np
import os
from keras.layers.core import K
K.set_learning_phase(False)

import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib


g = tf.GraphDef()
g.ParseFromString(open('outYOLO/frozen_YOLO_KERAS_RBC.pb', 'rb').read())
[n for n in g.node if n.name.find('input') != -1]
print 


with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            g, 
            input_map=None, 
            return_elements=None, 
            name="prefix", 
            op_dict=None, 
            producer_op_list=None
)



image = cv2.imread('BloodImage_00221.jpg')


input_image = cv2.resize(image, (416, 416))


input_image = input_image / 255.
input_image = input_image[:,:,::-1]

#plt.imshow(input_image); plt.show()

input_image = np.expand_dims(input_image, 0)
print(input_image.shape)


import argparse 
import tensorflow as tf

    # Let's allow the user to pass the filename as an argument
    
    # We use our "load_graph" function
#graph = load_graph('outYOLO/frozen_YOLO_KERAS.pb')

    # We can verify that we can access the list of operations in the graph
#for op in graph.get_operations():
    #print(op.name)
        # prefix/Placeholder/inputs_placeholder
        # ...
        # prefix/Accuracy/predictions
        
    # We access the input and output nodes 
x = graph.get_tensor_by_name('prefix/conv2d_1_input:0')
y = graph.get_tensor_by_name('prefix/reshape_1/Reshape:0')
        
    # We launch a Session
with tf.Session(graph=graph) as sess:
        # Note: we didn't initialize/restore anything, everything is stored in the graph_def
    y_out = sess.run(y, feed_dict={
        x: input_image # < 45
    })
        

image = interpret_netout(image, y_out[0])
plt.imshow(image[:,:,::-1]); plt.show()



