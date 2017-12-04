import numpy as np

import backend
import nn

class Model(object):
    """Base model class for the different applications"""
    def __init__(self):
        self.get_data_and_monitor = None
        self.learning_rate = 0.0

    def run(self, x, y=None):
        raise NotImplementedError("Model.run must be overriden by subclasses")

    def train(self):
        """
        Train the model.

        `get_data_and_monitor` will yield data points one at a time. In between
        yielding data points, it will also monitor performance, draw graphics,
        and assist with automated grading. The model (self) is passed as an
        argument to `get_data_and_monitor`, which allows the monitoring code to
        evaluate the model on examples from the validation set.
        """
        for x, y in self.get_data_and_monitor(self):
            graph = self.run(x, y)
            graph.backprop()
            graph.step(self.learning_rate)

class RegressionModel(Model):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_regression
        self.learning_rate = 0.01
        self.m = nn.Variable(1, 100)
        self.m2 = nn.Variable(100, 100)
        self.m3 = nn.Variable(100, 1)
        self.b = nn.Variable(100)
        self.b2 = nn.Variable(100)
        self.b3 = nn.Variable(1)

    def run(self, x, y=None):
        """
        Runs the model for a batch of examples.

        The correct outputs `y` are known during training, but not at test time.
        If correct outputs `y` are provided, this method must construct and
        return a nn.Graph for computing the training loss. If `y` is None, this
        method must instead return predicted y-values.

        Inputs:
            x: a (batch_size x 1) numpy array
            y: a (batch_size x 1) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 1) numpy array of predicted y-values
        """
        graph = nn.Graph([self.m, self.b, self.m2, self.b2, self.m3, self.b3])
        input_x = nn.Input(graph, x)
        #============= LAYER 01 ===============#
        xm = nn.MatrixMultiply(graph, input_x, self.m)
        xm_plus_b = nn.MatrixVectorAdd(graph, xm, self.b)
        #============= LAYER 02 ===============#
        relu = nn.ReLU(graph, xm_plus_b)
        xm2 = nn.MatrixMultiply(graph, relu, self.m2)
        xm_plus_b2 = nn.MatrixVectorAdd(graph, xm2, self.b2)
        #============= LAYER 03 ===============#
        relu2 = nn.ReLU(graph, xm_plus_b2)
        xm3 = nn.MatrixMultiply(graph, relu2, self.m3)
        xm_plus_b3 = nn.MatrixVectorAdd(graph, xm3, self.b3)

        if y is not None:
            input_y = nn.Input(graph, y)
            loss = nn.SquareLoss(graph, xm_plus_b3, input_y)
            return graph
        else:
            return graph.get_output(xm_plus_b3)


class OddRegressionModel(Model):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers.

    Unlike RegressionModel, the OddRegressionModel must be structurally
    constrained to represent an odd function, i.e. it must always satisfy the
    property f(x) = -f(-x) at all points during training.
    """
    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_regression
        self.learning_rate = 0.01
        self.m = nn.Variable(1, 100)
        self.m2 = nn.Variable(100, 100)
        self.m3 = nn.Variable(100, 1)
        self.b = nn.Variable(100)
        self.b2 = nn.Variable(100)
        self.b3 = nn.Variable(1)

    def run(self, x, y=None):
        """
        Runs the model for a batch of examples.

        The correct outputs `y` are known during training, but not at test time.
        If correct outputs `y` are provided, this method must construct and
        return a nn.Graph for computing the training loss. If `y` is None, this
        method must instead return predicted y-values.

        Inputs:
            x: a (batch_size x 1) numpy array
            y: a (batch_size x 1) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 1) numpy array of predicted y-values
        """
        graph = nn.Graph([self.m, self.b, self.m2, self.b2, self.m3, self.b3])
        input_x = nn.Input(graph, x)

        def hidden(x):
            #============= LAYER 01 ===============#
            xm = nn.MatrixMultiply(graph, x, self.m)
            xm_plus_b = nn.MatrixVectorAdd(graph, xm, self.b)
            #============= LAYER 02 ===============#
            relu = nn.ReLU(graph, xm_plus_b)
            xm2 = nn.MatrixMultiply(graph, relu, self.m2)
            xm_plus_b2 = nn.MatrixVectorAdd(graph, xm2, self.b2)
            #============= LAYER 03 ===============#
            relu2 = nn.ReLU(graph, xm_plus_b2)
            xm3 = nn.MatrixMultiply(graph, relu2, self.m3)
            return nn.MatrixVectorAdd(graph, xm3, self.b3)

        #============ ODD TRANSFORM ===========#
        neg_x = nn.Input(graph, x * np.array([[-1.0]]))
        gx = hidden(input_x)
        neg_gx = nn.MatrixMultiply(graph, hidden(neg_x), nn.Input(graph, np.array([[-1.0]])))
        fx = nn.MatrixVectorAdd(graph, gx, neg_gx)

        if y is not None:
            input_y = nn.Input(graph, y)
            loss = nn.SquareLoss(graph, fx, input_y)
            return graph
        else:
            return graph.get_output(fx)

class DigitClassificationModel(Model):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here.)
    """
    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_digit_classification
        self.learning_rate = 0.69
        self.m = nn.Variable(784, 400)
        self.m2 = nn.Variable(400, 10)
        self.b = nn.Variable(400)
        self.b2 = nn.Variable(10)

    def run(self, x, y=None):
        """
        Runs the model for a batch of examples.

        The correct labels are known during training, but not at test time.
        When correct labels are available, `y` is a (batch_size x 10) numpy
        array. Each row in the array is a one-hot vector encoding the correct
        class.

        Inputs:
            x: a (batch_size x 784) numpy array
            y: a (batch_size x 10) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 10) numpy array of scores (aka logits)
        """
        graph = nn.Graph([self.m, self.b, self.m2, self.b2])
        input_x = nn.Input(graph, x)
        #============= LAYER 01 ===============#
        xm = nn.MatrixMultiply(graph, input_x, self.m)
        xm_plus_b = nn.MatrixVectorAdd(graph, xm, self.b)
        #============= LAYER 02 ===============#
        relu = nn.ReLU(graph, xm_plus_b)
        xm2 = nn.MatrixMultiply(graph, relu, self.m2)
        xm_plus_b2 = nn.MatrixVectorAdd(graph, xm2, self.b2)

        if y is not None:
            input_y = nn.Input(graph, y)
            loss = nn.SoftmaxLoss(graph, xm_plus_b2, input_y)
            return graph
        else:
            return graph.get_output(xm_plus_b2)


class DeepQModel(Model):
    """
    A model that uses a Deep Q-value Network (DQN) to approximate Q(s,a) as part
    of reinforcement learning.
    """
    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_rl

        self.num_actions = 2
        self.state_size = 4
        self.learning_rate = 0.0015
        self.m = nn.Variable(4, 200)
        self.m2 = nn.Variable(200, 200)
        self.m3 = nn.Variable(200, 2)
        self.b = nn.Variable(200)
        self.b2 = nn.Variable(200)
        self.b3 = nn.Variable(2)

    def run(self, states, Q_target=None):
        """
        Runs the DQN for a batch of states.

        The DQN takes the state and computes Q-values for all possible actions
        that can be taken. That is, if there are two actions, the network takes
        as input the state s and computes the vector [Q(s, a_1), Q(s, a_2)]

        When Q_target == None, return the matrix of Q-values currently computed
        by the network for the input states.

        When Q_target is passed, it will contain the Q-values which the network
        should be producing for the current states.

        Inputs:
            states: a (batch_size x 4) numpy array
            Q_target: a (batch_size x 2) numpy array, or None
        Output:
            (if Q_target is not None) A nn.Graph instance, where the last added
                node is the loss
            (if Q_target is None) A (batch_size x 2) numpy array of Q-value
                scores, for the two actions
        """
        graph = nn.Graph([self.m, self.b, self.m2, self.b2, self.m3, self.b3])
        input_x = nn.Input(graph, states)
        #============= LAYER 01 ===============#
        xm = nn.MatrixMultiply(graph, input_x, self.m)
        xm_plus_b = nn.MatrixVectorAdd(graph, xm, self.b)
        #============= LAYER 02 ===============#
        relu = nn.ReLU(graph, xm_plus_b)
        xm2 = nn.MatrixMultiply(graph, relu, self.m2)
        xm_plus_b2 = nn.MatrixVectorAdd(graph, xm2, self.b2)
        #============= LAYER 03 ===============#
        relu2 = nn.ReLU(graph, xm_plus_b2)
        xm3 = nn.MatrixMultiply(graph, relu2, self.m3)
        xm_plus_b3 = nn.MatrixVectorAdd(graph, xm3, self.b3)

        if Q_target is not None:
            input_y = nn.Input(graph, Q_target)
            loss = nn.SquareLoss(graph, xm_plus_b3, input_y)
            return graph
        else:
            return graph.get_output(xm_plus_b3)

    def get_action(self, state, eps):
        """
        Select an action for a single state using epsilon-greedy.

        Inputs:
            state: a (1 x 4) numpy array
            eps: a float, epsilon to use in epsilon greedy
        Output:
            the index of the action to take (either 0 or 1, for 2 actions)
        """
        if np.random.rand() < eps:
            return np.random.choice(self.num_actions)
        else:
            scores = self.run(state)
            return int(np.argmax(scores))


class LanguageIDModel(Model):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here.)
    """
    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_lang_id

        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        self.learning_rate = 0.05
        self.d = 400

        self.m = nn.Variable(self.d, 400)
        self.m2 = nn.Variable(400, self.d)
        self.m3 = nn.Variable(400, 5)
        self.b = nn.Variable(400)
        self.b2 = nn.Variable(self.d)
        self.b3 = nn.Variable(5)

        self.h0 = nn.Variable(self.d)
        self.x = nn.Variable(self.num_chars, self.d)

    def run(self, xs, y=None):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        (batch_size x self.num_chars) numpy array, where every row in the array
        is a one-hot vector encoding of a character. For example, if we have a
        batch of 8 three-letter words where the last word is "cat", we will have
        xs[1][7,0] == 1. Here the index 0 reflects the fact that the letter "a"
        is the inital (0th) letter of our combined alphabet for this task.

        The correct labels are known during training, but not at test time.
        When correct labels are available, `y` is a (batch_size x 5) numpy
        array. Each row in the array is a one-hot vector encoding the correct
        class.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a (batch_size x self.num_chars) numpy array
            y: a (batch_size x 5) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 5) numpy array of scores (aka logits)
        """
        batch_size = xs[0].shape[0]

        #============= LAYER 00 ===============#
        graph = nn.Graph([self.m, self.b, self.m2, self.b2, self.m3, self.b3, self.h0, self.x])
        H0 = nn.Input(graph, np.zeros((batch_size, self.d)))
        Hi = nn.MatrixVectorAdd(graph, H0, self.h0)

        def f(H, x):
            input_x = nn.Input(graph, x)
            x_transform = nn.MatrixMultiply(graph, input_x, self.x)
            h_update = nn.MatrixVectorAdd(graph, H, x_transform)

            xm = nn.MatrixMultiply(graph, h_update, self.m)
            xm_plus_b = nn.MatrixVectorAdd(graph, xm, self.b)
            relu = nn.ReLU(graph, xm_plus_b)
            xm2 = nn.MatrixMultiply(graph, relu, self.m2)
            xm_plus_b2 = nn.MatrixVectorAdd(graph, xm2, self.b2)
            relu2 = nn.ReLU(graph, xm_plus_b2)
            return relu

        for i in xs:
            Hi = f(Hi, i)
        xm3 = nn.MatrixMultiply(graph, Hi, self.m3)
        xm_plus_b3 = nn.MatrixVectorAdd(graph, xm3, self.b3)

        if y is not None:
            input_y = nn.Input(graph, y)
            loss = nn.SoftmaxLoss(graph, xm_plus_b3, input_y)
            return graph
        else:
            return graph.get_output(xm_plus_b3)
