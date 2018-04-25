import math
import unittest
import functools
from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.legacy.optim as old_optim
import torch.nn.functional as F
from torch.optim import SGD
from torch.autograd import Variable
from torch import sparse
from test_common import TestCase, run_tests
import numpy as np
import p03_layers


def rosenbrock(tensor):
    x, y = tensor
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def drosenbrock(tensor):
    x, y = tensor
    return torch.DoubleTensor((-400 * x * (y - x ** 2) - 2 * (1 - x), 200 * (y - x ** 2)))


def wrap_old_fn(old_fn, **config):
    def wrapper(closure, params, state):
        return old_fn(closure, params, config, state)
    return wrapper


class TestOptim(TestCase):

    def _test_rosenbrock(self, constructor, old_fn):
        params_t = torch.Tensor([1.5, 1.5])
        state = {}

        params = Variable(torch.Tensor([1.5, 1.5]), requires_grad=True)
        optimizer = constructor([params])

        solution = torch.Tensor([1, 1])
        initial_dist = params.data.dist(solution)

        def eval():
            optimizer.zero_grad()
            loss = rosenbrock(params)
            loss.backward()
            # loss.backward() will give **slightly** different
            # gradients, than drosenbtock, because of a different ordering
            # of floating point operations. In most cases it doesn't matter,
            # but some optimizers are so sensitive that they can temporarily
            # diverge up to 1e-4, just to converge again. This makes the
            # comparison more stable.
            params.grad.data.copy_(drosenbrock(params.data))
            return loss

        for i in range(2000):
            optimizer.step(eval)
            old_fn(lambda _: (rosenbrock(params_t), drosenbrock(params_t)),
                   params_t, state)
            self.assertEqual(params.data, params_t)

        self.assertLessEqual(params.data.dist(solution), initial_dist)

    def _test_rosenbrock_sparse(self, constructor, sparse_only=False):
        params_t = torch.Tensor([1.5, 1.5])

        params = Variable(params_t, requires_grad=True)
        optimizer = constructor([params])

        if not sparse_only:
            params_c = Variable(params_t.clone(), requires_grad=True)
            optimizer_c = constructor([params_c])

        solution = torch.Tensor([1, 1])
        initial_dist = params.data.dist(solution)

        def eval(params, sparse_grad, w):
            # Depending on w, provide only the x or y gradient
            optimizer.zero_grad()
            loss = rosenbrock(params)
            loss.backward()
            grad = drosenbrock(params.data)
            # NB: We torture test the optimizer by returning an
            # uncoalesced sparse tensor
            if w:
                i = torch.LongTensor([[0, 0]])
                x = grad[0]
                v = torch.DoubleTensor([x / 4., x - x / 4.])
            else:
                i = torch.LongTensor([[1, 1]])
                y = grad[1]
                v = torch.DoubleTensor([y - y / 4., y / 4.])
            x = sparse.DoubleTensor(i, v, torch.Size([2]))
            if sparse_grad:
                params.grad.data = x
            else:
                params.grad.data = x.to_dense()
            return loss

        for i in range(2000):
            # Do cyclic coordinate descent
            w = i % 2
            optimizer.step(functools.partial(eval, params, True, w))
            if not sparse_only:
                optimizer_c.step(functools.partial(eval, params_c, False, w))
                self.assertEqual(params.data, params_c.data)

        self.assertLessEqual(params.data.dist(solution), initial_dist)

    def _test_basic_cases_template(self, weight, bias, input, constructor):
        weight = Variable(weight, requires_grad=True)
        bias = Variable(bias, requires_grad=True)
        input = Variable(input)
        optimizer = constructor(weight, bias)

        # to check if the optimizer can be printed as a string
        optimizer.__repr__()

        def fn():
            optimizer.zero_grad()
            y = weight.mv(input)
            if y.is_cuda and bias.is_cuda and y.get_device() != bias.get_device():
                y = y.cuda(bias.get_device())
            loss = (y + bias).pow(2).sum()
            loss.backward()
            return loss

        initial_value = fn().data[0]
        for i in range(200):
            optimizer.step(fn)
        self.assertLess(fn().data[0], initial_value)

    def _test_state_dict(self, weight, bias, input, constructor):
        weight = Variable(weight, requires_grad=True)
        bias = Variable(bias, requires_grad=True)
        input = Variable(input)

        def fn_base(optimizer, weight, bias):
            optimizer.zero_grad()
            i = input_cuda if weight.is_cuda else input
            loss = (weight.mv(i) + bias).pow(2).sum()
            loss.backward()
            return loss

        optimizer = constructor(weight, bias)
        fn = functools.partial(fn_base, optimizer, weight, bias)

        # Prime the optimizer
        for i in range(20):
            optimizer.step(fn)
        # Clone the weights and construct new optimizer for them
        weight_c = Variable(weight.data.clone(), requires_grad=True)
        bias_c = Variable(bias.data.clone(), requires_grad=True)
        optimizer_c = constructor(weight_c, bias_c)
        fn_c = functools.partial(fn_base, optimizer_c, weight_c, bias_c)
        # Load state dict
        state_dict = deepcopy(optimizer.state_dict())
        state_dict_c = deepcopy(optimizer.state_dict())
        optimizer_c.load_state_dict(state_dict_c)
        # Run both optimizations in parallel
        for i in range(20):
            optimizer.step(fn)
            optimizer_c.step(fn_c)
            self.assertEqual(weight, weight_c)
            self.assertEqual(bias, bias_c)
        # Make sure state dict wasn't modified
        self.assertEqual(state_dict, state_dict_c)

        # Check that state dict can be loaded even when we cast parameters
        # to a different type and move to a different device.
        if not torch.cuda.is_available():
            return

        input_cuda = Variable(input.data.float().cuda())
        weight_cuda = Variable(weight.data.float().cuda(), requires_grad=True)
        bias_cuda = Variable(bias.data.float().cuda(), requires_grad=True)
        optimizer_cuda = constructor(weight_cuda, bias_cuda)
        fn_cuda = functools.partial(fn_base, optimizer_cuda, weight_cuda, bias_cuda)

        state_dict = deepcopy(optimizer.state_dict())
        state_dict_c = deepcopy(optimizer.state_dict())
        optimizer_cuda.load_state_dict(state_dict_c)

        # Make sure state dict wasn't modified
        self.assertEqual(state_dict, state_dict_c)

        for i in range(20):
            optimizer.step(fn)
            optimizer_cuda.step(fn_cuda)
            self.assertEqual(weight, weight_cuda)
            self.assertEqual(bias, bias_cuda)

    def _test_basic_cases(self, constructor, ignore_multidevice=False):
        self._test_state_dict(
            torch.randn(10, 5),
            torch.randn(10),
            torch.randn(5),
            constructor
        )
        self._test_basic_cases_template(
            torch.randn(10, 5),
            torch.randn(10),
            torch.randn(5),
            constructor
        )
        # non-contiguous parameters
        self._test_basic_cases_template(
            torch.randn(10, 5, 2)[..., 0],
            torch.randn(10, 2)[..., 0],
            torch.randn(5),
            constructor
        )
        # CUDA
        if not torch.cuda.is_available():
            return
        self._test_basic_cases_template(
            torch.randn(10, 5).cuda(),
            torch.randn(10).cuda(),
            torch.randn(5).cuda(),
            constructor
        )
        # Multi-GPU
        if not torch.cuda.device_count() > 1 or ignore_multidevice:
            return
        self._test_basic_cases_template(
            torch.randn(10, 5).cuda(0),
            torch.randn(10).cuda(1),
            torch.randn(5).cuda(0),
            constructor
        )

    def _build_params_dict(self, weight, bias, **kwargs):
        return [dict(params=[weight]), dict(params=[bias], **kwargs)]

    def _build_params_dict_single(self, weight, bias, **kwargs):
        return [dict(params=bias, **kwargs)]

    def test_sgd(self):
        try:
            self._test_rosenbrock(
                lambda params: p03_layers.P3SGD(params, lr=1e-3),
                wrap_old_fn(old_optim.sgd, learningRate=1e-3)
            )
            self._test_rosenbrock(
                lambda params: p03_layers.P3SGD(params, lr=1e-3, momentum=0.9,
                                                dampening=0, weight_decay=1e-4),
                wrap_old_fn(old_optim.sgd, learningRate=1e-3, momentum=0.9,
                            dampening=0, weightDecay=1e-4)
            )
            self._test_basic_cases(
                lambda weight, bias: p03_layers.P3SGD([weight, bias], lr=1e-3)
            )
            self._test_basic_cases(
                lambda weight, bias: p03_layers.P3SGD(
                    self._build_params_dict(weight, bias, lr=1e-2),
                    lr=1e-3)
            )
            self._test_basic_cases(
                lambda weight, bias: p03_layers.P3SGD(
                    self._build_params_dict_single(weight, bias, lr=1e-2),
                    lr=1e-3)
            )
        except NotImplementedError:
            pass

class TestReluFunction(TestCase):

    def test_p3relu_function(self):

        #test general case
        relu_input = Variable(torch.Tensor(np.array([[1, 2, 3], [4, -5, 6]])))
        expected = Variable(torch.Tensor(np.array([[1, 2, 3], [4, 0, 6]])))

        self.assertNotEqual(relu_input, expected)
        self.assertEqual(p03_layers.p3relu(relu_input, False), expected)

        input2 = Variable(torch.Tensor(np.array([[-1, -1, -1], [-1, -5, -1]])))
        expected2 = Variable(torch.Tensor(np.array([[0, 0, 0], [0, 0, 0]])))

        self.assertNotEqual(input2, expected2)
        self.assertEqual(p03_layers.p3relu(input2), expected2)

        #inplace test
        p03_layers.p3relu(input2, False)
        self.assertNotEqual(input2, expected2)
        p03_layers.p3relu(input2, True)
        self.assertEqual(input2, expected2) 

class TestDropoutFunction(TestCase):

    def test_P3Dropout(self):
        # Test for when p > 1
        p_greater_than_1 = 1.5
        with self.assertRaises(ValueError):
            p03_layers.P3Dropout(p_greater_than_1, False)
            p03_layers.P3Dropout(p_greater_than_1, True)

        # Test for when p < 0
        p_less_than_0 = - 0.5
        with self.assertRaises(ValueError):
            p03_layers.P3Dropout(p_less_than_0, False)
            p03_layers.P3Dropout(p_less_than_0, True)

        # Test for p = 0.5
        dropout_input = torch.Tensor(np.array([[1, 2, 3], [4, -5, 6]]))
        expected = 0.5 * (dropout_input.size(0) * dropout_input.size(1)) # Number of zeros if p = 0.5 (p * length)

        for i in range (30):
            m = p03_layers.P3Dropout(0.5, False)
            # Expected number of zeros is greater than custom lower bound
            self.assertGreaterEqual(6 - (torch.nonzero(m(dropout_input)).size(0)), int(expected - 2))
            # Expected number of zeros is smaller than custom upper bound
            self.assertLessEqual(6 - (torch.nonzero(m(dropout_input)).size(0)), int(expected + 2))

class TestDropout2dFunction(TestCase):

    def test_P3Dropout2d(self):
        # Test for when p > 1
        p_greater_than_1 = 1.5
        with self.assertRaises(ValueError):
            p03_layers.P3Dropout(p_greater_than_1, False)
            p03_layers.P3Dropout(p_greater_than_1, True)

        # Test for when p < 0
        p_less_than_0 = - 0.5
        with self.assertRaises(ValueError):
            p03_layers.P3Dropout(p_less_than_0, False)
            p03_layers.P3Dropout(p_less_than_0, True)

        # Test for p = 0.5
        # We cannot exactly test since we always get random values
        # However, we can see if we are in fair range of the expected value
        # over a few loops.

        # Test for p = 0.5
        dropout2d_input = torch.randn(20, 16, 32, 32)
        prob = 0.5
        expected = prob * (dropout2d_input.size(0) * dropout2d_input.size(1)) # Number of zeros if p = 0.5 (p * length)
        # should get around 160, 32 x 32 channels zeroed out
        m = p03_layers.P3Dropout2d(prob, False)
        dropout_result = m(dropout2d_input)

        dropout2d_counter = 0
        for i in range(dropout2d_input.size(0)):
            for j in range(dropout2d_input.size(1)):
                # If channel is all zeroed out
                if (not np.count_nonzero(dropout_result[i][j].numpy())):
                    dropout2d_counter += 1
        # Check if number of zeroed out channels are within reasonable range
        self.assertGreaterEqual(dropout2d_counter, expected - 50)
        self.assertLessEqual(dropout2d_counter, expected + 50)


class TestReluClass(TestCase):

    def test_p3relu_class(self):
        #test general case
        relu_input = Variable(torch.Tensor(np.array([[1, 2, 3], [4, -5, 6]])))
        expected = Variable(torch.Tensor(np.array([[1, 2, 3], [4, 0, 6]])))
        m = p03_layers.P3ReLU()

        self.assertNotEqual(relu_input, expected)
        self.assertEqual(m(relu_input), expected)

        input2 = Variable(torch.Tensor(np.array([[-1, -1, -1], [-1, -5, -1]])))
        expected2 = Variable(torch.Tensor(np.array([[0, 0, 0], [0, 0, 0]])))

        self.assertNotEqual(input2, expected2)
        self.assertEqual(m(input2), expected2)

        #inplace test
        m = p03_layers.P3ReLU(True)
        self.assertNotEqual(input2, expected2)
        m(input2)
        self.assertEqual(input2, expected2) 

class TestLinearClass(TestCase):

    def test_p3linear(self):
        #General Test for working
        model = p03_layers.P3Linear(20,30)
        test_input = Variable(torch.randn(128, 20))
        output = model(test_input)
        m, n = output.shape

        self.assertNotEqual(output, test_input)
        self.assertEqual(m, 128)
        self.assertEqual(n , 30)

        #compare result with pytorch module Linear
        pytorch_model = nn.Linear(20,30)
        d = pytorch_model.state_dict()
        model_weight = Variable(d['weight'])
        model_bias = Variable(d['bias'])
        linear_model = p03_layers.P3LinearFunction()

        pytorch_model_output = pytorch_model(test_input)
        m_linear_model_output = linear_model(test_input, model_weight, model_bias)
        self.assertEqual(pytorch_model_output.shape, m_linear_model_output.shape)
        self.assertEqual(pytorch_model_output, m_linear_model_output)

        #check bias works or not
        m_linear_model_output = linear_model(test_input, model_weight)
        self.assertNotEqual(pytorch_model_output, m_linear_model_output)

        #model without bias
        test_input = Variable(torch.randn(128, 20))
        pytorch_model = nn.Linear(20, 10, bias=False)
        model_weight = Variable(pytorch_model.weight.data)
        linear_model = p03_layers.P3LinearFunction()

        pytorch_model_output = pytorch_model(test_input)
        m_linear_model_output = linear_model(test_input, model_weight)
        self.assertEqual(pytorch_model_output.shape, m_linear_model_output.shape)
        self.assertEqual(pytorch_model_output, m_linear_model_output)

class TestELUClass(TestCase):

    def test_p3elu(self):
        #General working test
        m = p03_layers.P3ELU()
        test_input = Variable(torch.randn(5,5))
        output = m(test_input)

        self.assertNotEqual(test_input, output)

        #compare result with torch module
        torch_module = nn.ELU()
        torch_ouput = torch_module(test_input)
        self.assertEqual(output.shape, torch_ouput.shape)
        self.assertEqual(output, torch_ouput)

        #different alpha size
        m = p03_layers.P3ELU(alpha=2, inplace=False)
        output = m(test_input)
        torch_module = nn.ELU(alpha=2, inplace=False)
        torch_ouput = torch_module(test_input)
        self.assertEqual(output.shape, torch_ouput.shape)
        self.assertEqual(output, torch_ouput)

        #different large test input
        test_input = Variable(torch.randn(128,128))
        m = p03_layers.P3ELU(alpha=2, inplace=False)
        output = m(test_input)
        torch_module = nn.ELU(alpha=2, inplace=False)
        torch_ouput = torch_module(test_input)
        self.assertEqual(output.shape, torch_ouput.shape)
        self.assertEqual(output, torch_ouput)

        # inplace parameter check
        # m = p03_layers.P3ELU(alpha=2, inplace=True)
        # output = m(test_input)
        # self.assertEqual(output.shape, torch_ouput.shape)
        # self.assertEqual(output, torch_ouput)

def check_net(model):
    model.train()
    # output from network
    data = torch.autograd.Variable(torch.rand(2, 1, 28, 28))
    # output from network
    target = torch.autograd.Variable((torch.rand(2) * 2).long())
    optimizer = SGD(model.parameters(), lr=0.1)
    optimizer.zero_grad()
    # Forward prediction step
    output = model(data)
    loss = F.nll_loss(output, target)
    # Backpropagation step
    loss.backward()
    optimizer.step()

def test_nets():
    model = p03_layers.Net()
    check_net(model)

def test_sgd():
    return TestOptim


def test_p3dropout():
    return TestDropoutFunction


def test_p3dropout2d():
    return TestDropout2dFunction


def test_p3linear():
    return TestLinearClass


def test_p3relu_function():
    return TestReluFunction


def test_p3relu_class():
    return TestReluClass


def test_p3elu():
    return TestELUClass


def test_p3bce_loss():
    # TODO Implement me
    pass


if __name__ == '__main__':
    # Automatically call every function
    # in this file which begins with test.
    # see unittest library docs for details.
    run_tests()
