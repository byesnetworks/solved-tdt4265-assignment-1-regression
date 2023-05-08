Download Link: https://assignmentchef.com/product/solved-tdt4265-assignment-1-regression
<br>
<h1>Task 1: Regression</h1>

<strong>Notation: </strong>We use index <em>k </em>to represent a node in the output layer, index <em>j </em>to represent a node in the hidden layer, and index <em>i </em>to represent an input unit <em>x<sub>i</sub></em>. Hence, the weight from node <em>i </em>in the input layer to node <em>k </em>in the output layer is <em>w<sub>ki</sub></em>. We write the activation of output unit <em>k </em>as ˆ<em>y<sub>k </sub></em>= <em>f</em>(<em>z<sub>k</sub></em>), where <em>f </em>represents the output unit activation function (sigmoid for logistic regression or softmax for softmax regression). In equations where multiple training examples are used (for example summing over samples), we will use <em>n </em>to specify which training example we are referring to. Hence, <em>y<sub>k</sub><sup>n </sup></em>is the output of node <em>k </em>for training example <em>n</em>. If we do not specify <em>n </em>by writing <em>y<sub>k</sub></em>, we implicitly refer to <em>y<sub>k</sub><sup>n</sup></em>. Capital letters N, I, J or K refer to the total number of nodes in a layer. For logistic regression we will not specify which output node we’re referring to, as there is only a single output node <em>k</em>. Therefore weight <em>w<sub>k,i </sub></em>can be written as <em>w<sub>i</sub></em>.

<h2>Logistic Regression</h2>

Logistic regression is a simple tool to perform binary classification. Logistic regression can be modeled as using a single neuron reading a input vector <em>x </em>∈ R<em><sup>I</sup></em><sup>+<a href="#_ftn1" name="_ftnref1">[1]</a> </sup>and parameterized by a weight vector <em>w </em>∈ R<em><sup>I</sup></em><sup>+1</sup>. <em>I </em>is the number of input nodes, and we add a 1 at the beginning for a bias parameter (this is known as the ”bias trick”). The neuron outputs the probability that <em>x </em>is a member of class <em>C</em><sub>1</sub>. This can be written as,

(1)

<em>P</em>(<em>x </em>∈ <em>C</em><sub>2</sub>|<em>x</em>) = 1 − <em>P</em>(<em>x </em>∈ <em>C</em><sub>1</sub>|<em>x</em>) = 1 − <em>f</em>(<em>x</em>)                                                             (2)

where <em>f</em>(<em>x</em>) returns the probability of <em>x </em>being a member of class <em>C</em><sub>1</sub>; <em>f </em>∈ [0<em>,</em>1] <sup>1</sup>. By defining the output of our network as ˆ<em>y</em>, we have ˆ<em>y </em>= <em>f</em>(<em>x</em>).

We use the <strong>cross entropy loss </strong>function (Equation 3) for two categories to measure how well our function performs over our dataset. This loss function measures how well our hypothesis function <em>f </em>does over the <em>N </em>data points.

<em>,           </em>where <em>C<sup>n</sup></em>(<em>w</em>) = −(<em>y<sup>n </sup></em>ln(ˆ<em>y<sup>n</sup></em>) + (1 − <em>y<sup>n</sup></em>)ln(1 − <em>y</em>ˆ<em><sup>n</sup></em>))                               (3)

Here, <em>y<sup>n </sup></em>is the target value (also known as the label of the image). Note that we are computing the average cost function, such that the magnitude of our cost function is not dependent on number of training examples. Our goal is to minimize this cost function through gradient descent, such that the cost function reaches a minimum of 0. This happens when <em>y<sup>n </sup></em>= <em>y</em>ˆ<em><sup>n </sup></em>for all <em>n</em>.

<h2>Softmax Regression</h2>

Softmax regression is simply a generalization of logistic regression to multi-class classification. Given an input <em>x </em>which can belong to <em>K </em>different classes, softmax regression will output a vector ˆ<em>y </em>(with length <em>K</em>), where each element ˆ<em>y<sub>k </sub></em>represents the probability that <em>x </em>is a member of class <em>k</em>.

<em>I</em>

<em>,           </em>where <em>z<sub>k </sub></em>= <em>w<sub>k</sub><sup>T </sup></em>· <em>x </em>= <sup>X</sup><em>w<sub>k,i </sub></em>· <em>x<sub>i                                                                                 </sub></em>(4)

<em>i</em>

Equation 4 is known as the Softmax function and it has the attribute that = 1. Note that now <em>w </em>is no longer a vector, but a weight matrix, <em>w </em>∈ <strong>R</strong><em><sup>K</sup></em><sup>×<em>I</em></sup>.

The cross-entropy cost function for multiple classes is defined as,

)                                                 (5)

<strong>For this task, please:</strong>

<ul>

 <li>[0<em>.</em>275<em>pt</em>] Derive the gradient for Logistic Regression. To minimize the cost function with gradient descent, we require the gradient of the cost function. Show that for Equation 3, the gradient is:</li>

</ul>

(6)

Show thorough work such that your approach is clear.

<em>Hint: </em>To solve this, you have to use the chain rule. Also, you can use the fact that:

))                                                             (7)

<ul>

 <li>[0<em>.</em>375<em>pt</em>] Derive the gradient for Softmax Regression. For the multi-class cross entropy cost in Equation 5, show that the gradient is:</li>

</ul>

)                                                                  (8)

A few hints if you get stuck:

<ul>

 <li>Derivation of the softmax is the hardest part. Break it down into two cases.</li>

 <li></li>

 <li></li>

</ul>

<h1>Task 2: Logistic Regression through Gradient Descent</h1>

In this assignment you are going to start classifying digits in the well-known dataset MNIST. The MNIST dataset consists of 70<em>,</em>000 handwritten digits, split into 10 object classes (the numbers 0-9). The images are 28×28 grayscale images, and every image is perfectly labeled. The images are split into two datasets, a training set consisting of 60<em>,</em>000 images, and a testing set consisting of 10<em>,</em>000 images. For this assignment, we will use a subset of the MNIST dataset<a href="#_ftn2" name="_ftnref2"><sup>[2]</sup></a>.

<strong>Bias trick: </strong>Each image is 28×28, so the unraveled vector will be <em>x </em>∈ R<sup>784</sup>. For each image, append a ’1’ to it, giving us <em>x </em>∈ R<sup>785</sup>. With this trick, we don’t need to implement the forward and backward pass for the bias.

<strong>Logistic Regression through gradient descent</strong>

For this task, we will use mini-batch gradient descent to train a logistic regression model to predict if an image is either a 2 or a 3. We will remove all images in the MNIST dataset that are not a 2 or 3 (this pre-processing is already implemented in the starter code). The target is 1 if the the input is from the ”2” category, and 0 otherwise.

<strong>Mini-batch gradient descent </strong>is a method that takes a batch of images to compute an average gradient, then use this gradient to update the weights. Use the gradient derived for logistic regression to classify <em>x </em>∈ R<sup>785 </sup>for the two categories 2’s and 3’s.

<strong>Vectorizing code: </strong>We recommend you to vectorize the code with numpy, which will make the runtime of your code significantly faster. Note that vectorizing your code is not required, but highly recommended (it will be required for assignment 2). Vectorizing it simply means that if you want to, for example, compute the gradient in Equation 6, you can compute it in one go instead of iterating over the number of examples and weights. For example, <em>w<sup>T</sup>x </em>can be witten as w.dot(x).

<strong>For this task, please:</strong>

<ul>

 <li>[0<em>.</em>5<em>pt</em>] Before implementing our gradient descent training loop, we will implement a couple of essential functions. Implement four functions in the file py.

  <ul>

   <li>Implement a function that pre-processes our images in the function pre_process_images. This should normalize our images from the range [0<em>,</em>255] to [−1<em>,</em>1] <a href="#_ftn3" name="_ftnref3"><sup>[3]</sup></a>, and it should apply the bias trick. Implement a function that performs the forward pass through our single layer neural network. Implement this in the function forward. This should implement the network outlined by Equation 1.</li>

   <li>Implement a function that performs the backward pass through our single layer neural network. Implement this in the function backward. To find the gradient for our weight, we can use the equation derived in task 1 (Equation 6).</li>

   <li>Implement cross entropy loss in the function cross_entropy_loss. This should compute the average of the cross entropy loss over all targets/labels and predicted outputs. The cross entropy loss is shown in Equation 3.</li>

  </ul></li>

</ul>

We have included a couple of simple tests to help you debug your code. This also includes a gradient approximation test that you should get working. For those interested, this is explained in more detail in the Appendix.

<strong>Note that you should not start on the subsequent tasks before all tests are passing!</strong>

<ul>

 <li>[0<em>.</em>35<em>pt</em>] Implement logistic regression with mini-batch gradient descent for a single layer neural network. The network should consist of a single weight matrix with 784 + 1 inputs and a single output (the matrix will have shape 785×1). Initialize the weights (before any training) to all zeros. We’ve set the default hyperparameters for you, so there is no need to change these.</li>

</ul>

During training, track the training loss for each gradient step (this is implemented in the starter code). Less frequently, track the validation loss over the whole validation set (in the starter code, this is tracked every time we progress 20% through the training set).

Implement this in the function train_step in task2.py.

<strong>(report) </strong>In your report, include a plot of the training and validation loss over training. Have the number of gradient steps on the x-axis, and the loss on the y-axis. Use the ylim function to zoom in on the graph (for us, ylim([0, 0.2]) worked fine).

<ul>

 <li>[0<em>.</em>15<em>pt</em>] Implement a function that computes the binary classification accuracy<sup>4 </sup>over a dataset. Implement this in the function calculate_accuracy.</li>

</ul>

<strong>(report) </strong>Compute the accuracy on the training set and validation set over training. Plot this in a graph (similar to the loss), and include the plot in your report. Use the ylim function to zoom in on the graph (for us, ylim([0.93, 0.99]) worked fine).

<strong>Early Stopping: </strong>Early stopping is a tool to stop the training before your model overfits on the training dataset. By using a validation set along with our training set<a href="#_ftn4" name="_ftnref4"><sup>[4]</sup></a>, we can regularly check if our model is starting to overfit or not. If we notice that the cost function on the validation set stops to improve, we can stop the training and return the weights at the minimum validation loss.

<strong>Dataset shuffling: </strong>Shuffling the training dataset between each epoch improves convergence. By using shuffling you present a new batch of examples each time which the network has never seen, which will produce larger errors and improve gradient descent <a href="#_ftn5" name="_ftnref5"><sup>[5]</sup></a>.

<ul>

 <li>[0<em>.</em>15<em>pt</em>] Implement early stopping into your training loop. Use the following early stop criteria: stop the training if the validation loss does not improve after passing through 20% of the training dataset 10 times. Increase the number of epochs to 500. <strong>(report) </strong>After how many epochs does early stopping kick in?</li>

</ul>

You can implement early stopping in the file trainer.py.

<ul>

 <li>[0<em>.</em>2<em>pt</em>] Implement dataset shuffling for your training. Before each epoch, you should shuffle all the samples in the dataset. Implement this in the function batch_loader in py</li>

</ul>

<strong>(report) </strong>Include a plot in your report of the validation accuracy with and without shuffle. You should notice that the validation accuracy has less ”spikes”. Why does this happen?

4accuracy = Number of correct predictions. The prediction is determined as 1 if ˆ<em>y </em>≥ 0<em>.</em>5 else 0 Total number of predictions

<h1>Task 3: Softmax Regression through Gradient Descent</h1>

In this task, we will perform a 10-way classification on the MNIST dataset with softmax regression. Use the gradient derived for softmax regression loss and use mini-batch gradient descent to optimize your model

<strong>One-hot encoding: </strong>With multi-class classification tasks it is required to one-hot encode the target values. Convert the target values from integer to one-hot vectors. (E.g: 3 → [0<em>,</em>0<em>,</em>0<em>,</em>1<em>,</em>0<em>,</em>0<em>,</em>0<em>,</em>0<em>,</em>0<em>,</em>0]). The length of the vector should be equal to the number of classes (<em>K </em>= 10 for MNIST, 1 class per digit). <strong>For this task, please:</strong>

<ul>

 <li>[0<em>.</em>55<em>pt</em>] Before implementing our gradient descent training loop, we will implement a couple of essential functions. Implement four functions in the file py.

  <ul>

   <li>Implement a function that one-hot encodes our labels in the function one_hot_encode. This should return a new vector with one-hot encoded labels. Implement a function that performs the forward pass through our single layer softmax model. Implement this in the function forward. This should implement the network outlined by Equation 4.</li>

   <li>Implement a function that performs the backward pass through our single layer neural network. Implement this in the function backward. To find the gradient for our weight, use Equation 8.</li>

   <li>Implement cross entropy loss in the function cross_entropy_loss. This should compute the average of the cross entropy loss over all targets/labels and predicted outputs. The cross entropy loss is defined in Equation 5.</li>

  </ul></li>

</ul>

We have included a couple of simple tests to help you debug your code.

<ul>

 <li>[0<em>.</em>1<em>pt</em>] <strong>The rest of the task 3 subtasks should be implemented in </strong>py<strong>.</strong></li>

</ul>

Implement softmax regression with mini-batch gradient descent for a single layer neural network. The network should consist of a single weight matrix, with 784 + 1 inputs and ten outputs (shape 785 × 10). Initialize the weight (before any training) to all zeros.

Implement this in train_step in task3.py. This function should be identical to the task2b, except that you are using a different cross entropy loss function.

<strong>(report) </strong>In your report, include a plot of the training and validation loss over training. Have the number of gradient steps on the x-axis, and the loss on the y-axis. Use the ylim function to zoom in on the graph (for us, ylim([0.2, .6]) worked fine).

<ul>

 <li>[0<em>.</em>15<em>pt</em>] Implement a function that computes the multi-class classification accuracy over a dataset. Implement this in the function calculate_accuracy.</li>

</ul>

<strong>(report) </strong>Include in your report a plot of the training and validation accuracy over training.

<ul>

 <li>[0<em>.</em>15<em>pt</em>] <strong>(report) </strong>For your model trained in task 3c, do you notice any signs of overfitting? Explain your reasoning.</li>

</ul>

<h1>Task 4: Regularization</h1>

One way to improve generalization is to use regularization. Regularization is a modification we make to a learning algorithm that is intended to reduce its generalization error <a href="#_ftn6" name="_ftnref6"><sup>[6]</sup></a>. Regularization is the idea that we should penalize the model for being too complex. In this assignment, we will carry this out by introducing a new term in our objective function to make the model ”smaller” by minimizing the weights.

<em>J</em>(<em>w</em>) = <em>C</em>(<em>w</em>) + <em>λR</em>(<em>w</em>)<em>,                                                                             </em>(9)

where <em>R</em>(<em>w</em>) is the complexity penalty and <em>λ </em>is the strength of regularization (constant). There are several forms for <em>R</em>, such as <em>L</em><sub>2 </sub>regularization

<em>,                                                                    </em>(10)

where <em>w </em>is the weight vector of our model.

<strong>For your report, please:</strong>

<ul>

 <li>[0<em>.</em>15<em>pt</em>] <strong>(report) </strong>Derive the update term for softmax regression with <em>L</em><sub>2 </sub>regularization, that is, find , where <em>C </em>is given by Equation 5.</li>

 <li>[0<em>.</em>3<em>pt</em>] Implement <em>L</em><sub>2 </sub>regularization in your backward pass. You can implement the regularization in backward in py.</li>

</ul>

<strong>For the remaining part of the assignment, you can implement the functionality in the file task3.py or create a new file.</strong>

<strong>(report) </strong>Train two different models with different <em>λ </em>values for the <em>L</em><sub>2 </sub>regularization. Use <em>λ </em>= 0<em>.</em>0 and <em>λ </em>= 2<em>.</em>0. Visualize the weight for each digit for the two models. Why are the weights for the model with <em>λ </em>= 2<em>.</em>0 less noisy?

The visualization should be similar to Figure 1.

Figure 1: The visualization of the weights for a model with <em>λ </em>= 0<em>.</em>0 (top row), and <em>λ </em>= 2<em>.</em>0 (bottom row).

<ul>

 <li>[0<em>.</em>2<em>pt</em>] <strong>(report) </strong>Train your model with different values of <em>λ</em>: 2<em>.</em>0, 0<em>.</em>2, 0<em>.</em>02, 0<em>.</em> Note that for each value of <em>λ</em>, you should train a new network from scratch. Plot the validation accuracy for different values of <em>λ </em>on the same graph (during training). Have the accuracy on the y-axis, and number of training steps on the x-axis.</li>

 <li>[0<em>.</em>2<em>pt</em>] <strong>(report) </strong>You will notice that the validation accuracy degrades when applying any amount of regularization. What do you think is the reason for this?</li>

 <li>[0<em>.</em>2<em>pt</em>] <strong>(report) </strong>Plot the length (<em>L</em><sub>2 </sub>norm, ||<em>w</em>||<sup>2</sup>) of the weight vector for the each <em>λ </em>value in task 4c. What do you observe? Plot the <em>λ </em>value on the x-axis and the <em>L</em><sub>2 </sub>norm on the y-axis.</li>

</ul>

Note that you should plot the <em>L</em><sub>2 </sub>norm of the weight <strong>after </strong>each network is finished training.

<h1>Appendix</h1>

<h2>Gradient Approximation test</h2>

When implementing neural networks from the bottom up, there can occur several minor bugs that completely destroy the training process. Gradient approximation is a method to get a numerical approximation to what the gradient should be, and this is extremely useful when debugging your forward, backward, and cost function. If the test is incorrect, it indicates that there is a bug in one (or more) of these functions.

It is possible to compute the gradient with respect to one weight by using numerical approximation:

<em>,                                                            </em>(11)

where <em>ϵ </em>is a small constant (e.g. 10<sup>−2</sup>), and <em>C</em>(<em>w<sub>w</sub></em><em><sub>ij </sub></em>+ <em>ϵ</em>) refers to the error on example <em>x<sup>n </sup></em>when weight <em>w<sub>ji </sub></em>is set to <em>w<sub>ji </sub></em>+ <em>ϵ</em>. The difference between the gradients should be within big-O of <em>ϵ</em><sup>2</sup>, so if <em>ϵ </em>= 10<sup>−2</sup>, your gradients should agree within <em>O</em>(10<sup>−4</sup>).

If your gradient approximation does not agree with your calculated gradient from backpropagation, there is something wrong with your code!

<a href="#_ftnref1" name="_ftn1">[1]</a> The function <em>f </em>is known as the sigmoid activation function

<a href="#_ftnref2" name="_ftn2">[2]</a> We will only use the first 20<em>,</em>000 images in the training set to reduce computation time.

<a href="#_ftnref3" name="_ftn3">[3]</a> Normalizing the input to be zero-centered improves convergence for neural networks. Read more about this in Lecun et al. <a href="https://scholar.google.com/scholar?cluster=2656844239423157571">Efficient Backprop</a> Section 4.3.

<a href="#_ftnref4" name="_ftn4">[4]</a> Note that we never train on the validation set.

<a href="#_ftnref5" name="_ftn5">[5]</a> For those interested, you can read more about dataset shuffling in Section 4.2 in <a href="https://scholar.google.com/scholar?cluster=2656844239423157571">Efficient Backprop</a><a href="https://scholar.google.com/scholar?cluster=2656844239423157571">.</a>

<a href="#_ftnref6" name="_ftn6">[6]</a> The generalization error can be thought of as training error – validation error.