# Keras or TensorFlow? - An Incomplete Introductory Guide In Both APIs

Keras or TensorFlow? Which one should I use? This blog post describes the differences between the [Keras API](https://keras.io/) against the [TensorFlow (TF) API](https://www.tensorflow.org/) for Python, my language of choice for machine learning. In particular, I will focus on the most important features that are needed to get started into deep learning. At the end, you should be hopefully able to make a decision yourself which one to use.

Most of the features will be demonstrated with the help of a simple linear regression problem. Please note that I will not be able to cover every topic in detail or some of the features that you might think are important may be not in there. But this post is rather kept _simplistic_ and should serve as an overview. I will give pointers to more advanced/additional topics - that’s why it is incomplete. Moreover, this is not a typical tool versus another tool blog post. I don’t like to compare oranges against apples if you understand what I mean;)

## General

**Keras:**

* Keras is a lightweight high-level wrapper around numerical computation libraries ([called backend](https://keras.io/backend/)) dedicated to deep learning such as TensorFlow and Theano
* It is open source and written in Python (supports 2.7-3.5)
* It was originally created by François Chollet ([@fchollet](https://twitter.com/fchollet))
* Keras is now integrated into TensorFlow


**TensorFlow:**

* TensorFlow is an open source software library for numerical computation that uses data flow graphs (many machine learning models, in particular neural networks, can be visualized via directed graphs)
* Competitors are  for example Torch, Theano, and Caffe
* TF was originally developed by Google’s Brain team
* It supports Python, C++, Haskell, Java and Go
* In TensorFlow there are two important concepts:
    1. Everything is a tensor which is an n-dimensional matrix
        * 0-d tensor: scalar
        * 1-d tensor: vector
        * 2-d tensor: matrix
        * ...
    2. Lazy evaluation of the computational graph (a series of TF operations arranged into a graph of nodes) i.e. computation is separated from execution
* A TF core program consists of two sections:
    1. Building the computational graph
    2. Running the computational graph

## Installation

**Keras:**

```sh
pip install keras
```

**TensorFlow:**

```sh
pip install tensorflow
```

## Import

**Keras:**

```python
import keras
```

**TensorFlow:**

```python
import tensorflow as tf
```

## Building the Model

**Keras:**

We can easily create linear to complex models through the [`sequential`](https://keras.io/models/sequential/) API, which stacks up layers in a linear fashion. For linear regression, we can write this as follow:

```python
from keras.models import Sequential
from keras.layers.core import Dense

model = Sequential()
model.add(Dense(units=1, activation="linear", input_dim=1))
```

* Keras includes many commonly used layers:
    - Regular `dense` layers for feed-forward neural networks
    - `Dropout`, `normalization` and `noise` layers to prevent overfitting and improve learning
    - Common `convolutional` and `pooling layers` (max and average) for CNNs
    - `Flatten` layers to add fully connected layers after convolutional nets
    - `Embedding` layers for Word2Vec problems
    - `Recurrent` layers (simpleRNN, GRU and LSTM) for RNNs
    - `Merge` layers to combine more than one neural net
    - Many more… you can also [write your own Keras layers](https://keras.io/layers/writing-your-own-keras-layers/)
* Many common activation functions are available like `relu`, `tanh` and `sigmoid` depending on the problem that you like to solve (read [“Yes you should understand backprop” by Andrej Karpathy](https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b) if you want to understand the effect of backpropagation on some activation functions)
* The activation function can also be passed through an `Activation` layer instead of the `activation` argument

Alternatively if you prefer one-liners, we could have also done something like this:

```python
model = Sequential([Dense(units=1, activation="linear", input_dim=1)])
```

Or we could have used [Keras’s functional API](https://keras.io/getting-started/functional-api-guide/):

```python
from keras.models import Model
from keras.layers import Dense, Input

X = Input(shape=(1,))
Y = Dense(units=1, activation ="linear")(X)

model = Model(inputs=X, outputs=Y)
```

Then we need to configure the learning settings, which is done via the `compile` step. For linear regression it makes sense to use [mean square error](https://en.wikipedia.org/wiki/Mean_squared_error) to evaluate the quality of our estimated model, `loss="mean_squared_error"` (other [loss functions](https://keras.io/losses/) like `cross-entropy` are also available out-of-the-box).

We will also use the standard settings for [stochastic gradient descent (SGD)](https://en.wikipedia.org/wiki/Stochastic_gradient_descent), `optimizer="sgd"`

```python
model.compile(loss="mean_squared_error", optimizer="sgd")
```

If you want different settings for the optimizers then do this:

``` python
from keras.optimizers import SGD
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="mean_squared_error", optimizer=sgd)
```

Keras supports [most common optimizers](https://keras.io/optimizers/) like RMSprop, Adagrad and many others. An excellent overview of different optimization algorithms and what effect they have while training a neural network can be found by this [blog article of Sebastian Ruder](http://sebastianruder.com/optimizing-gradient-descent/index.html#gradientdescentoptimizationalgorithms).

**TensorFlow:**

In TF, we first need to build up the computational graph. We define the inputs first:

```python
X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

W = tf.Variable(tf.random_normal(shape=[]), name="weight")
b = tf.Variable(tf.random_normal(shape=[]), name="bias")
```

* A [placeholder](https://www.tensorflow.org/api_docs/python/tf/placeholder) is a tensor where values are provided later
* [Variables](https://www.tensorflow.org/api_docs/python/tf/Variable) are tensors which allow us to add trainable parameters to a graph
* There are also [constants, sequences and random tensors](https://www.tensorflow.org/api_guides/python/constant_op)

Then we need to define the linear model, the loss function and the optimizer. We will use the same loss (`mse`) and optimizer (`sgd`) as in the Keras case.

```python
Y_predicted = tf.add(tf.multiply(X, W), b)
cost = tf.losses.mean_squared_error(labels=Y, predictions=Y_predicted)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
```

* The way how we define the type/number of layers along with its activation is very different in TF than in Keras - it is much more explicit
* It is a good practice to wrap up all the placeholders, variables, constants, model definitions etc in a `Graph` class especially if you want to use multiple graphs in the same process
```python
g = tf.Graph()
with g.as_default():
    # Define the computational graph
    ...
```
* It also has many common used layers like Keras out-of-the box, see [`tf.layers`](https://www.tensorflow.org/api_docs/python/tf/layers) and [`tf.contrib.layers`](https://www.tensorflow.org/api_docs/python/tf/contrib/layers)
* TF also supports a wide range of [loss functions](https://www.tensorflow.org/api_docs/python/tf/losses) and [optimizers](https://www.tensorflow.org/api_guides/python/train#Optimizers) like Keras

## Training the Model

**Keras:**

```python
model.fit(x, y)
```

In the standard output we get a progress bar which shows the training progress (full gradient update), the loss at each epoch and the epoch itself:

```sh
Epoch 1/10
100/100 [==============================] - 0s - loss: 1.1310
Epoch 2/10
100/100 [==============================] - 0s - loss: 0.5461
Epoch 3/10
100/100 [==============================] - 0s - loss: 0.2656
...
```

We can also control the size of the batch per gradient update (common sizes are 32, 64, 128, 256...) through `batch_size`.

```python
model.fit(x, y, batch_size=1)
```

We can also easily evaluate our model with `validation_split` which holds a given percentage between 0 and 1 of the data back.

```python
model.fit(x, y, batch_size=1, validation_split=0.2)

Train on 80 samples, validate on 20 samples
Epoch 1/10
80/80 [==============================] - 0s - loss: 0.9673 - val_loss: 0.6618
Epoch 2/10
80/80 [==============================] - 0s - loss: 0.5153 - val_loss: 0.3498
Epoch 3/10
80/80 [==============================] - 0s - loss: 0.2726 - val_loss: 0.1853
...
```

Finally, we can also use [`callback`](https://keras.io/callbacks/) which is set of functions that can used during training. Important functions are:

* `keras.callbacks.History()` - Recording the loss history, `loss` and `val_loss` if `validation_split` is set
* `keras.callbacks.ModelCheckpoint()` - Saving the model after each epoch
* `keras.callbacks.EarlyStopping()` - Stopping training early depending on the monitored metric such as `val_loss`
* `keras.callbacks.TensorBoard()` - Storing the log for Tensorboard

**TensorFlow:**

For training the model we need to run the computational graph via a [`Session`](https://www.tensorflow.org/api_docs/python/tf/Session) object. Below is the most simplistic way to do this which prints the training loss after each epoch:

```python
# Parameters
epochs = 10
learning_rate = 0.01

# Start the session and initialized variables
sess = tf.Session()
sess.run(tf.global_variables_initializer()) # Variables needs to be initialized

# Train the model for n epochs
for epoch in range(epochs):
    print("Epoch: {}/{}".format(epoch+1, epochs))
    for i, j in zip(x, y): # batch_size=1
        _, loss = sess.run([optimizer, cost], feed_dict={X: i, Y: j})
    print("loss : {}".format(loss))

print(sess.run([W, b]))
sess.close()
```

* Functionalities like early stopping, a Keras-like progress bar for training or even validation split have to be implemented; though some of the features like early stopping are available via `tf.contrib` but those are experimental code
* It is also a good practice to start and close the `Session` with a `with` statement
```python
with tf.Session(graph=g) as sess:
    # Run the computational graph
    ...
```

### Reproducibility

* To get reproducible results during the training process in TensorFlow only it makes sense to fix the [graph-level random seed](https://www.tensorflow.org/api_docs/python/tf/set_random_seed) by setting `tf.set_random_seed(...)` for all operations or for a specific one (Please note that random seed only affects the active graph only)
* For Keras while using TF as backend, we need to fix both numpy’s and TF’s seed:

``` python
import numpy as np
np.random.seed(...)
import tensorflow as tf
tf.set_random_seed(...)
```

## Saving the Model

**Keras:**

Save the model with `model.save()` and reload the model with `load_model()`:

```python
from keras.models import load_model

# save model
model.save("./model.h5")

# load model
model_loaded = load_model("./model.h5")
```

* `pip install h5py` before saving the model as Keras uses [HDF5](https://support.hdfgroup.org/HDF5/) to store its models and this doesn’t come with installing Keras
* You can also [save the model architecture (JSON/YAML) and model weights separately](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model)

**TensorFlow:**

TensorFlow models can be saved via the `tf.train.Saver()` object. Usually what you do is:

```python
g = tf.Graph()
with g.as_default():
    # Define the computational graph
    ...

    # Initialize the variables
    init = tf.global_variables_initializer()

    # Add saver to store the model
    saver = tf.train.Saver()

with tf.Session(graph=g) as sess:
    sess.run(init)
    # Run the computational graph
    ...

    # Save the session in a file
    save_path = saver.save(sess, "./model.ckpt")
```

Restoring the model:

```python
g = tf.Graph()
with g.as_default():
    # Recreate the computational graph
    ...

    # Add saver to load the model
    saver = tf.train.Saver()

with tf.Session(graph=g) as sess:
    # Restore the model
    saver.restore(sess, "./model.ckpt")
```

* TF’s `Saver` actually saves an [intermediate state (called checkpoint)](https://www.tensorflow.org/programmers_guide/meta_graph) of the trained model (weights, the graph and its metadata for different timesteps)
* Have a look at [this StackOverFlow post](http://stackoverflow.com/questions/33759623/tensorflow-how-to-save-restore-a-model) if you want to restore your model without defining the graph again
* TF uses [Protocol Buffers](https://developers.google.com/protocol-buffers/?hl=en) (`protobuf`) to [save all its files to disk](https://www.tensorflow.org/extend/tool_developers/)
* [`freeze_graph`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py) can be used to save the model into a single file - this is important when serving a model in production as we don’t need any special metadata files along with the model or if you want to port the models to other languages such as Java, C++ etc… (read [Morgan Giraud’s post](https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc) for an excellent tutorial on how to freeze a TF model and then serve it as python API with flask)

## Model Information

**Keras:**

A useful feature is `model.summary()` which shows the number of used layers, output shape and number of trainable parameters (alternatively `model.get_config()` can be used to get information of the model as well).

```python
model.summary()
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense_9 (Dense)              (None, 1)                 2
=================================================================
Total params: 2
Trainable params: 2
Non-trainable params: 0
_________________________________________________________________
```

* After the model is trained it is also useful to see the model weights `model.get_weights()`
* Keras also provides tools to visualize your models through [`keras.utils.vis_utils`](https://keras.io/visualization/) (Note: [`pydot`](https://github.com/erocarrera/pydot) is needed `pip install pydot`)
* Through `callback`, we can also use Tensorboard

**TensorFlow:**

TF includes [Tensorboard](https://www.tensorflow.org/get_started/summaries_and_tensorboard) - a visualization tool that can be used to better understand, debug and optimize TF programs. Here is a simple example just to record the model graph:

```python
g = tf.Graph()
with g.as_default():
    # Define the computational graph
    ...

with tf.Session(graph=g) as sess:
    writer = tf.summary.FileWriter("./log_folder/", sess.graph)
    # Run the computational graph
    ...

    writer.close()
```

* It is useful to use the [`summary`](https://www.tensorflow.org/api_guides/python/summary) operations to get more explicit information about the model:
    - Use the [`tf.summary.scalar`](https://www.tensorflow.org/api_docs/python/tf/summary/scalar) operations to record for example the learning rate and loss
    - To visualize the distribution of weights or bias, you could use [`tf.summary.histogram`](https://www.tensorflow.org/api_docs/python/tf/summary/histogram)
* If the neural network is large, i.e. has many nodes, it would make sense to organize logically related operations into groups using [`tf.name_scope`](https://www.tensorflow.org/api_docs/python/tf/name_scope)
* A more advanced example can be [found on the TF page](https://www.tensorflow.org/get_started/summaries_and_tensorboard)

To run TensorBoard, use this command:

```sh
python -m tensorflow.tensorboard --logdir="./log_folder/"
```

Once it is running, go to your web browser (`localhost:6006`) to view TensorBoard.

## Prediction

**Keras:**

```python
model.predict(x_test)
```

We can use `model.predict(x_test)` to generate predictions (real values in terms of regression and probabilities for classification) or we could use `model.predict_classes(x_test)` to get the class in a classification problem directly.

**TensorFlow:**

Similar to saving the model, prediction happens in the `session`:

```python
g = tf.Graph()
with g.as_default():
    # Define the computational graph
    ...

    # Placeholders & variables
    ...

    # Linear model
    Y_predicted = tf.add(tf.multiply(X, W), b)

    # Cost function and optimizer
    ...

with tf.Session(graph=g) as sess:
    # Run the computational graph
    ...

    # Train the model
    ...

    # Make prediction
    prediction = sess.run(Y_predicted, feed_dict={X: x_test})
    print(prediction)
```

For classification problems in TF, we need to take the `arg_max` of the `tf.nn.softmax(Y_predicted)` function to get the predicted classes.

## Some Other Model Examples

### Logistic Regression

**Keras:**
```python
model = Sequential()
model.add(Dense(num_classes, activation="softmax", input_shape=(num_features,)))
model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
```

**TensorFlow:**
```python
g = tf.Graph()
with g.as_default():
    X = tf.placeholder(tf.float32, shape=[None, num_features])
    Y = tf.placeholder(tf.float32, shape=[None, num_classes])
    W = tf.Variable(tf.zeros([num_features, num_classes]))
    b = tf.Variable(tf.zeros([num_classes]))

    Y_predicted = tf.add(tf.matmul(X, W),  b)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=Y_predicted))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_predicted, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
```

### Multilayer Perceptron (MLP)

**Keras:**
```python
model = Sequential()
model.add(Dense(64, activation="relu", input_shape=(num_features,)))
model.add(Dropout(0.5))
model.add(Dense(32, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
```

**TensorFlow:**
```python
g = tf.Graph()
with g.as_default():
    def init_weights(shape):
        weights = tf.random_normal(shape, stddev=0.1)
        return tf.Variable(weights)

    X = tf.placeholder(tf.float32, shape=[None, num_features])
    Y = tf.placeholder(tf.float32, shape=[None, num_classes])
    w_1 = init_weights((num_features, 64))
    w_2 = init_weights((64, 32))
    w_3 = init_weights((32, num_classes))
    keep_prob = tf.constant(0.5, tf.float32)

    def mlp(X, w_1, w_2, w_3):
        layer_1 = tf.nn.relu(tf.matmul(X, w_1))
        layer_1_drop = tf.nn.dropout(layer_1, keep_prob)
        layer_2 = tf.nn.relu(tf.matmul(layer_1_drop, w_2))
        layer_2_drop = tf.nn.dropout(layer_2, keep_prob)
        out_layer = tf.matmul(layer_2_drop, w_3)
        return out_layer

    Y_predicted = mlp(X, w_1, w_2, w_3)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=Y_predicted))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_predicted, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
```

### Convolutional Neural Network (CNN)

**Keras:**
```python
model = Sequential()
model.add(Conv2D(32, (5, 5), strides=(1, 1), activation="relu", input_shape=(width, height, img_dim), padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
model.add(Conv2D(64, (5, 5), strides=(1, 1), activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
model.add(Flatten())
model.add(Dense(1024, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
```

**TensorFlow:**
```python
g = tf.Graph()
with g.as_default():
    X = tf.placeholder(tf.float32, shape=[None, width, height, img_dim])
    Y = tf.placeholder(tf.float32, shape=[None, num_classes])
    keep_prob = tf.constant(0.5, tf.float32)

    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    h_conv1 = tf.nn.relu(conv2d(X, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, num_classes])
    b_fc2 = bias_variable([num_classes])

    Y_predicted = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=Y_predicted))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    correct_prediction = tf.equal(tf.argmax(Y_predicted, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
```

## Useful Helpers

### Datasets

Usually, it would be very convenient to have some built-in datasets so that we can jump start some examples...

**Keras:**

* Classification: CIFAR-10, CIFAR-100, MNIST, IMDB movie reviews, Reuters newswire
* Regression: Boston housing price

More information can be found on [the Keras page](https://keras.io/datasets/#datasets).

**TensorFlow:**

TF provides a couple of datasets but they are spread everywhere:

* Some of the datasets can be found in the [`learn` contribution module](https://www.tensorflow.org/get_started/tflearn), e.g. the Iris and Boston housing price dataset, MNIST etc..
* Through [TF-Slim](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim) (also part of the contribution module) which is another lightweight high-level API for TF but developed by TensorFlow’s team, we can also get [additional datasets](https://github.com/tensorflow/models/tree/master/slim) such as Flowers, CIFAR-10, MNIST and ImageNet

### Pre-Trained Models

For many image classification problem, we normally don’t train the models from scratch (because it is computationally expensive or we don’t have much data) but we often start from pre-trained models and fine-tune it...

**Keras:**

* Keras has [several pre-trained models](https://keras.io/applications/#available-models) e.g. Xception, VGG16, VGG19 etc.., that are trained on ImageNet

**TensorFlow:**

* For TF, pre-trained models exist but are not in-built e.g. some of the pre-trained models can [be accessed via TF-Slim](https://github.com/tensorflow/models/tree/master/slim#Pretrained)

### Others

**Keras:**

* Preprocessing tools for [sequences](https://keras.io/preprocessing/sequence/), [text](https://keras.io/preprocessing/text/) and [image](https://keras.io/preprocessing/image/) data (the image preprocessing is awesome as you can easily augment image data with a number of random transformations e.g. rotation, zoom etc... - [see this post from the Keras author](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html) for a detailed use case)
* Other useful features:
    - One-hot encoding of target variables: `from keras.utils.np_utils import to_categorical`
    - Normalize vector: `from keras.utils.np_utils import normalize`

**TensorFlow:**

* TensorFlow has [TensorFlow Serving](https://tensorflow.github.io/serving/) which is used to operationalize TF models in production (an interesting [use case from Zendesk](https://medium.com/zendesk-engineering/how-zendesk-serves-tensorflow-models-in-production-751ee22f0f4b) using TensorFlow Serving)
* It provides a specialized debugger called [`tfdbg`](https://www.tensorflow.org/programmers_guide/debugger) to debug your TF program (also check out the [slides from Jongwook Choi](https://wookayin.github.io/tensorflow-talk-debugging/#1) (must read) to get more information on how/where to use it)
* [Threading and queues](https://www.tensorflow.org/programmers_guide/threading_and_queues) are supported for asynchronous computation (Morgan Giraud provides a [very cool example](https://blog.metaflow.fr/tensorflow-how-to-optimise-your-input-pipeline-with-queues-and-multi-threading-e7c3874157e0) on how to optimize the read data step)

## Conclusion

We only covered the surface of both APIs. There are many more features that we haven’t talked about. There are pros and cons for both APIs. Keras is particularly made for fast prototyping and it definitely serves its purpose. TensorFlow, on the other hand is much more verbose but you get a higher degree of flexibility and control. There are other wrappers around TensorFlow. For example, we highlighted some features from TF-Slim because it is in-built. Another interesting one is [TF Learn](http://tflearn.org/). Its syntax is quite nice and it has a very good documentation. As you can see there are many options. At the end of the day you have to choose yourself what suits your problem the best.

#### So what is your favorite language of choice for deep learning? Tell me why?

## Some Other Useful Links

* A [Complete Guide](https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html) To Using Keras as Part of a TensorFlow Workflow - François Chollet
* [Learning Deep Learning with Keras](http://p.migdal.pl/2017/04/30/teaching-deep-learning.html) (Excellent overview for someone who wants to get started into Deep Learning in general) - Piotr Migdal
* TensorFlow in a Nutshell: Part [One](https://medium.com/@camrongodbout/tensorflow-in-a-nutshell-part-one-basics-3f4403709c9d), [Two](https://chatbotnewsdaily.com/tensorflow-in-a-nutshell-part-two-hybrid-learning-98c121d35392), [Three](https://hackernoon.com/tensorflow-in-a-nutshell-part-three-all-the-models-be1465993930) - Camron Godbout
* [Some (clean) TensorFlow Examples](https://github.com/aymericdamien/TensorFlow-Examples) by the Author of TF Learn - Aymeric Damien
* [TensorFlow for Deep Learning Research](http://web.stanford.edu/class/cs20si/) (check out the slides and Github repo) - Chip Huyen
* [MetaFlow AI](https://blog.metaflow.fr/) (check out their TensorFlow best practice series - it's amazing) - Morgan Giraud & Thomas Olivier
