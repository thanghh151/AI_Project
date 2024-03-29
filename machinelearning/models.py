import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(self.w, x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        prod = nn.as_scalar(self.run(x))
        return (prod >= 0) - (prod < 0)

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        batch_size = 1
        change_flag = True
        while change_flag:
            change_flag = False
            for x, y in dataset.iterate_once(batch_size):
                result = self.get_prediction(x)
                if result != nn.as_scalar(y):
                    self.w.update(nn.Constant(nn.as_scalar(y)*x.data), 1)
                    change_flag = True

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.lr = .01
        self.w1 = nn.Parameter(1, 128)
        self.b1 = nn.Parameter(1, 128)
        self.w2 = nn.Parameter(128, 64)
        self.b2 = nn.Parameter(1, 64)
        self.w3 = nn.Parameter(64, 1)
        self.b3 = nn.Parameter(1, 1)
        self.params = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        first_layer = nn.ReLU(nn.AddBias(nn.Linear(x, self.w1),\
                                self.b1))
        second_layer = nn.ReLU(nn.AddBias(nn.Linear(first_layer, self.w2),\
                                self.b2))
        output_layer = nn.AddBias(nn.Linear(second_layer, self.w3),\
                               self.b3)
        return output_layer

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        y_hat = self.run(x)
        return nn.SquareLoss(y_hat, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        batch_size = 50
        loss = float('inf')
        while loss >= .015:
            for x, y in dataset.iterate_once(batch_size):
                loss = self.get_loss(x, y)
                print(nn.as_scalar(loss))
                grads = nn.gradients(loss, self.params)
                loss = nn.as_scalar(loss)
                for i in range(len(self.params)):
                    self.params[i].update(grads[i], -self.lr)

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.lr = .01
        self.w1 = nn.Parameter(1, 128)
        self.b1 = nn.Parameter(1, 128)
        self.w2 = nn.Parameter(128, 64)
        self.b2 = nn.Parameter(1, 64)
        self.w3 = nn.Parameter(64, 1)
        self.b3 = nn.Parameter(1, 1)
        self.params = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        first_layer = nn.ReLU(nn.AddBias(nn.Linear(x, self.w1),\
                                self.b1))
        second_layer = nn.ReLU(nn.AddBias(nn.Linear(first_layer, self.w2),\
                                self.b2))
        output_layer = nn.AddBias(nn.Linear(second_layer, self.w3),\
                               self.b3)
        return output_layer

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        y_hat = self.run(x)
        return nn.SquareLoss(y_hat, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        batch_size = 50
        loss = float('inf')
        while loss >= .015:
            for x, y in dataset.iterate_once(batch_size):
                loss = self.get_loss(x, y)
                print(nn.as_scalar(loss))
                grads = nn.gradients(loss, self.params)
                loss = nn.as_scalar(loss)
                for i in range(len(self.params)):
                    self.params[i].update(grads[i], -self.lr)

class DigitClassificationModel(object):
    def __init__(self):
        self.batch_size = 0
        self.in_dimension = 784
        self.out_dimension = 10
        self.alpha = -0.01
        self.setting = 1

        self.layer1_dimension = 100
        self.w_1 = nn.Parameter(self.in_dimension, self.layer1_dimension)
        self.b_1 = nn.Parameter(1, self.layer1_dimension)

        if self.setting == 1:
            self.layer2_dimension = 70
            self.w_2 = nn.Parameter(self.layer1_dimension, self.layer2_dimension)
            self.b_2 = nn.Parameter(1, self.layer2_dimension)

            self.layer3_dimension = self.out_dimension
            self.w_3 = nn.Parameter(self.layer2_dimension, self.layer3_dimension)
            self.b_3 = nn.Parameter(1, self.layer3_dimension)
        else:
            self.layer2_dimension = self.out_dimension
            self.w_2 = nn.Parameter(self.layer1_dimension, self.layer2_dimension)
            self.b_2 = nn.Parameter(1, self.layer2_dimension)

    def run(self, x):
        if self.batch_size == 0:
            self.batch_size = x.data.shape[0]

        if self.setting == 1:
            first_layer = nn.ReLU(nn.AddBias(nn.Linear(x, self.w_1), self.b_1))
            second_layer = nn.ReLU(nn.AddBias(nn.Linear(first_layer, self.w_2), self.b_2))
            return nn.AddBias(nn.Linear(second_layer, self.w_3), self.b_3)
        else:
            first_layer = nn.ReLU(nn.AddBias(nn.Linear(x, self.w_1), self.b_1))
            second_layer = nn.ReLU(nn.AddBias(nn.Linear(first_layer, self.w_2), self.b_2))
            return second_layer

    def get_loss(self, x, y):
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        while True:
            n = 0
            for x, y in dataset.iterate_once(self.batch_size):
                n += 1
                loss = self.get_loss(x, y)
                origin = [self.w_1, self.b_1, self.w_2, self.b_2] if self.setting == 2 else [self.w_1, self.b_1, self.w_2, self.b_2, self.w_3, self.b_3]
                grad = nn.gradients(loss, origin)
                for i in range(len(origin)):
                    origin[i].update(grad[i], self.alpha)
            if dataset.get_validation_accuracy() > 0.96:
                self.alpha = -0.003
            if dataset.get_validation_accuracy() > 0.972:
                break
class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        # Khởi tạo ma trận trọng số cho lớp thứ nhất
        self.w1 = nn.Parameter(self.num_chars, 128)
        # Khởi tạo ma trận trọng số cho lớp thứ hai
        self.w2 = nn.Parameter(128, 128)
        # Khởi tạo ma trận trọng số cho lớp thứ hai
        self.w3 = nn.Parameter(128, len(self.languages))
        # Khởi tạo vector bias cho lớp thứ ba
        self.b3 = nn.Parameter(1, len(self.languages))
        
        # Thiết lập tốc độ học
        self.lr = 0.1
        # Thiết lập kích thước batch
        self.batch_size = 100

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        # Khởi tạo trạng thái của mạng
        h = nn.ReLU(nn.Linear(xs[0], self.w1))
        # Duyệt qua các ký tự trong danh sách `xs` và cập nhật trạng thái của mạng
        for x in xs[1:]:
            h = nn.ReLU(nn.Add(nn.Linear(x, self.w1), nn.Linear(h, self.w2)))
        # Đưa trạng thái hiện tại của mạng qua một lớp tuyến tính và một lớp cộng bias để tính toán khả năng của mỗi ký tự thuộc về từng ngôn ngữ
        y = nn.AddBias(nn.Linear(h, self.w3), self.b3)
        return y



    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        # Trả về một node loss
        return nn.SoftmaxLoss(self.run(xs), y)


    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while True:
            # Lặp vô hạn để huấn luyện mô hình
            for x, y_ in dataset.iterate_forever(self.batch_size):
                # Tính toán loss
                loss = self.get_loss(x, y_)
                # Kiểm tra điều kiện dừng
                if dataset.get_validation_accuracy() > 0.85:
                    return
                
                # Tính gradient của loss đối với các tham số
                gradients = nn.gradients(loss, [self.w1, self.w2, self.w3, self.b3])
                params = [self.w1, self.w2, self.w3, self.b3]
                # Cập nhật các tham số bằng phương pháp gradient descent
                for param, gradient in zip(params, gradients):
                    param.update(gradient, -self.lr)