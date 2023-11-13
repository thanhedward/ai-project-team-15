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
        # DotProduct chỉ được sử dụng với perceptron
        return nn.DotProduct(x, self.w)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """

        res = 1 if nn.as_scalar(nn.DotProduct(self.w, x)) >= 0 else -1
        return res


    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        while True:
            match = True
            for x, y in dataset.iterate_once(1):
                if nn.as_scalar(y) != self.get_prediction(x):
                    match = False
                    self.w.update(x, nn.as_scalar(y))
            if match:
                break

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.learning_rate = 0.05
        # Tham số ở tầng 1
        self.weight_1 = nn.Parameter(1, 256) # Trọng số
        self.bias_1 = nn.Parameter(1, 256) # Độ lệch
        # Tham số ở tầng 2
        self.weight_2 = nn.Parameter(256, 256) # Trọng số
        self.bias_2 = nn.Parameter(1, 256) # Độ lệch
        # Tham số ở tầng 3
        self.weight_3 = nn.Parameter(256, 1) # Trọng số
        self.bias_3 = nn.Parameter(1, 1) # Độ lệch
        # Danh sách các tham số
        self.params = [self.weight_1, self.bias_1, self.weight_2, self.bias_2, self.weight_3, self.bias_3]

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        # Tính tích vô hướng
        first_hidden_layer_linear = nn.Linear(x, self.weight_1) 
        # Cộng tích vô hướng với độ lệch để được đầu vào cho hàm kích hoạt
        first_hidden_layer_input = nn.AddBias(first_hidden_layer_linear, self.bias_1)
        # Tính đầu ra của hàm kích hoạt hàm ReLU (hàm thay thế các số âm thành 0) 
        first_hidden_layer = nn.ReLU(first_hidden_layer_input) 
        # Tính tích vô hướng
        second_hidden_layer_linear = nn.Linear(first_hidden_layer, self.weight_2)
        # Cộng tích vô hướng với độ lệch để được đầu vào cho hàm kích hoạt 
        second_hidden_layer_input = nn.AddBias(second_hidden_layer_linear, self.bias_2) 
        # Tính đầu ra của hàm kích hoạt hàm ReLU (hàm thay thế các số âm thành 0)
        second_hidden_layer = nn.ReLU(second_hidden_layer_input) 
        # Đầu ra dự đoán
        output_layer = nn.AddBias(nn.Linear(second_hidden_layer, self.weight_3),self.bias_3) 

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
        y_predict = self.run(x) # Giá trị dự đoán của y
        return nn.SquareLoss(y_predict, y) # Tính loss giữa y dự đoán và y thực tế

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        batch_size = 200
        while True:
            for x, y in dataset.iterate_once(batch_size):
            # Tính toán giá trị mất mát
                loss = self.get_loss(x, y) 
            # Tính toán gradient ("một vector chứa đạo hàm riêng") của hàm mất mát
                grads = nn.gradients(loss, self.params) 
            # Trích xuất từ loss một số trong Python để so sánh
                loss = nn.as_scalar(loss) 
            # Tạo vòng lặp để điều chỉnh các tham số weight và bias để loss nhỏ nhất
                for i in range(len(self.params)): 
            # Trừ dần giá trị tích của gradient và tốc độ học và cập nhật giá trị để tiến gần với loss bé nhất
                    self.params[i].update(grads[i], -self.learning_rate) 
            # Khi loss bé hơn 0.01("đạt yêu cầu") thì dừng vòng lặp
            if loss < 0.01: 
                break           


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"

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

        # Tốc độ học
        self.learning_rate = 0.1

        # Initialize your model parameters here 
        # Tất cả các từ trong cùng 1 batch đều có cùng độ dài
        self.initial_w = nn.Parameter(self.num_chars, 128) # Trọng số của input
        self.initial_b = nn.Parameter(1, 128) # Độ lệch

        # Hidden layers
        self.weight = nn.Parameter(128, 128)
        self.bias = nn.Parameter(1, 128)

        # output
        self.output_w = nn.Parameter(128, len(self.languages)) #self.languages = 5
        self.output_b =  nn.Parameter(1, len(self.languages))
        "*** YOUR CODE HERE ***"

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
        
        # Lý thuyết trong hướng dẫn: h1 = f(initial)(x0) ~ z0 = x0 * W0
        # Xử lí kí tự đầu tiên
        z0 = nn.Linear(xs[0], self.initial_w)
        z0 = nn.ReLU(nn.AddBias(z0, self.initial_b))
        z = z0

        # Lấy đầu ra của phần tử trước làm đầu vào của phần tử sau
        # Lý thuyết trong hướng dẫn hi = f(hi, xi) ~ zi = xi * Wx + hi * Whidden
        for i in range(1, len(xs)):
            # Lí thuyết trong hướng dẫn z = nn.Add(nn.Linear(x, W), nn.Linear(h, W_hidden))
            z_raw = nn.Add(nn.Linear(xs[i], self.initial_w), nn.Linear(z, self.weight))
            z = nn.ReLU(nn.AddBias(z_raw, self.bias))
        return nn.AddBias(nn.Linear(z, self.output_w), self.output_b)
    



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
        output_predict = self.run(xs) # Giá trị dự đoán của xs
        return nn.SoftmaxLoss(output_predict, y) # Tính loss giữa output dự đoán và output thực tế

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        batch_size = 100
        loss = None
        accuracy = 0
        # Tăng giá trị dừng vòng lặp để cải thiện độ chính xác
        while accuracy < 0.85: 
            for x, y in dataset.iterate_once(batch_size):
                loss = self.get_loss(x, y)
                grads = nn.gradients(loss, [self.initial_w, self.weight, self.output_w, self.initial_b, self.bias, self.output_b]) # Tính gradient
                loss = nn.as_scalar(loss)
                # Trừ đi tích của gradient và learning_rate để tiến dần đến cực trị cho mất mát là nhỏ nhất
                self.initial_w.update(grads[0], -self.learning_rate)
                self.weight.update(grads[1], -self.learning_rate)
                self.output_w.update(grads[2], -self.learning_rate)
                self.initial_b.update(grads[3], -self.learning_rate)
                self.bias.update(grads[4], -self.learning_rate)
                self.output_b.update(grads[5], -self.learning_rate)
            accuracy = dataset.get_validation_accuracy()
