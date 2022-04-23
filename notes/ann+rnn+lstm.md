# Neural networks & deep learning concepts for NLP
## ANN
An artificial neural network (ANN) is computational model that produces outputs from inputs based on data inputted, processed, produced and transmitted by each node in a network of interconnected nodes (links can be unidirectional or bidirectional).

The data from each node is either transmitted to other nodes within the network, or is part of the final output. The data to each node is either aggregated based on the outputs and weights of the links of connected nodes, or is part of the initial input. ANN is inspired from biological neural networks (BNN) such as brains, and forms the basis for deep learning algorithms.

## Deep learning
Deep learning is a subfield of machine learning. It refers to the usage of an ANN using more than one layer, hence the term 'deep', which refers to depth of network layers. Multiple network layers enhances the ANN's ability to adapt to more diverse and complex problems, since there are many more nodes, hence many more link weights and link structures to modify according to requirements. The focus of deep learning is on automating the processing and structuring of unstructured data. In particular, it automates feature extraction.

## RNN
A recurrent neural network (RNN) is a type of ANN that uses the output of the hidden layers as an additional input to the new input in a sequence of inputs. Hence, RNN deals with sequential data or time series data, where past values (at or up to a certain lag ) are associated to present values.

In particular, RNN's are used in NLP. The meaning of each word in a sentence depends on the previous words, since the order of words affects meaning to some degree. Similarly, the meaning of each sentence may depend on preceding sentences. However, a standard RNN have short-term memory. In order to analyse entire texts, this memory must be extended.

## LSTM
A long short-term memory network (LSTM) is an extension of RNN that extends the memory of RNN. Hence, it can learn from past data at much greater lags. This is the neural network we will use in sentiment analysis.

## Necessary concepts
### Loss function 
Loss function (also called error function) for a model computes the distance between the current output of the model and the expected output (provided in the training data). Hence, this function is used during the training of the model to evaluate the performance of the model, hence guide changes to the model (in case of neural networks, these are changes to weights and connections.

In the context of a neural network, loss function can be interpreted as a function of all the weights in the neural network. Furthermore, it can be interpreted as a composite function, each layer's functional expression being applied to the previous layer's functional expression.

### Gradient descent (GD) 
Gradient descent is an iterative first-order  optimisation algorithm used to find a local minimum or local maximum of a given function. In machine learning, the function under consideration is the loss function, and we aim to minimize this loss function's output by trying to find the global minimum by finding the local minimum using a certain step value .

### Backpropagation 
Backpropagation is a term that can be expanded to " backward propagation of errors". It is an optimization algorithm for machine learning models (particularly for this that apply ANN's) that uses gradient descent. Given an ANN and a loss function, the method calculates the gradient (rate of change) of the loss function with respect to the neural network's weights (i.e. the neural network's weights are considered as the factor variables for the loss function), layer-by-layer.

"Backwards" refers to the way the algorithm calculates the gradient of the loss function with respect to the the neural network's ultimate layer first, the penultimate layer second, and so on until the first layer. This is significant since computations of the gradient for one layer (which is a partial derivative, since the loss function is assumed to depend on all the weights of the neural network) are reused in the computation of the gradient for the previous layer. The reason for reusing gradients in this manner becomes clear when considering the loss function as a composite function where each layer's functional expression being applied to the previous layer's functional expression. Hence, the gradient for the loss function can be calculated using the chain rule.