from __future__ import print_function
import os
import argparse
import datetime
import six
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.optim as optim
import torchvision
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
from tqdm import tqdm

import numpy as np

from tensorboardX import SummaryWriter
import callbacks

# Training settings
parser = argparse.ArgumentParser(description='Deep Learning JHU Assignment 1 - Fashion-MNIST')
parser.add_argument('--batch-size', type=int, default=256, metavar='B',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='TB',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='E',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--optimizer', type=str, default='sgd', metavar='O',
                    help='Optimizer options are sgd, p3sgd, adam, rms_prop')
parser.add_argument('--momentum', type=float, default=0.5, metavar='MO',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=100, metavar='I',
                    help="""how many batches to wait before logging detailed
                            training status, 0 means never log """)
parser.add_argument('--dataset', type=str, default='mnist', metavar='D',
                    help='Options are mnist and fashion_mnist')
parser.add_argument('--data_dir', type=str, default='../data/', metavar='F',
                    help='Where to put data')
parser.add_argument('--name', type=str, default='', metavar='N',
                    help="""A name for this training run, this
                            affects the directory so use underscores and not spaces.""")
parser.add_argument('--model', type=str, default='default', metavar='M',
                    help="""Options are default, P2Q7DefaultChannelsNet,
                    P2Q7HalfChannelsNet, P2Q7DoubleChannelsNet,
                    P2Q8BatchNormNet, P2Q9DropoutNet, P2Q10DropoutBatchnormNet,
                    P2Q11ExtraConvNet, P2Q12RemoveLayerNet, and P2Q13UltimateNet.""")
parser.add_argument('--print_log', action='store_true', default=False,
                    help='prints the csv log when training is complete')

required = object()


def timeStamped(fname, fmt='%Y-%m-%d-%H-%M-%S_{fname}'):
    """ Add a timestamp to your training run's name.
    """
    # http://stackoverflow.com/a/5215012/99379
    return datetime.datetime.now().strftime(fmt).format(fname=fname)

# choose the dataset


def prepareDatasetAndLogging(args):
    # choose the dataset
    if args.dataset == 'mnist':
        DatasetClass = datasets.MNIST
    elif args.dataset == 'fashion_mnist':
        DatasetClass = datasets.FashionMNIST
    else:
        raise ValueError('unknown dataset: ' + args.dataset + ' try mnist or fashion_mnist')

    training_run_name = timeStamped(args.dataset + '_' + args.name)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    # Create the dataset, mnist or fasion_mnist
    dataset_dir = os.path.join(args.data_dir, args.dataset)
    training_run_dir = os.path.join(args.data_dir, training_run_name)
    train_dataset = DatasetClass(
        dataset_dir, train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_dataset = DatasetClass(
        dataset_dir, train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # Set up visualization and progress status update code
    callback_params = {'epochs': args.epochs,
                       'samples': len(train_loader) * args.batch_size,
                       'steps': len(train_loader),
                       'metrics': {'acc': np.array([]),
                                   'loss': np.array([]),
                                   'val_acc': np.array([]),
                                   'val_loss': np.array([])}}
    if args.print_log:
        output_on_train_end = os.sys.stdout
    else:
        output_on_train_end = None

    callbacklist = callbacks.CallbackList(
        [callbacks.BaseLogger(),
         callbacks.TQDMCallback(),
         callbacks.CSVLogger(filename=training_run_dir + training_run_name + '.csv',
                             output_on_train_end=output_on_train_end)])
    callbacklist.set_params(callback_params)

    tensorboard_writer = SummaryWriter(log_dir=training_run_dir, comment=args.dataset + '_embedding_training')

    # show some image examples in tensorboard projector with inverted color
    images = 255 - test_dataset.test_data[:100].float()
    label = test_dataset.test_labels[:100]
    features = images.view(100, 784)
    tensorboard_writer.add_embedding(features, metadata=label, label_img=images.unsqueeze(1))
    return tensorboard_writer, callbacklist, train_loader, test_loader


def _assert_no_grad(variable):
    assert not variable.requires_grad, ("nn criterions don't compute the gradient w.r.t. targets - please "
                                        "mark these variables as not requiring gradients")


class _Loss(nn.Module):
    def __init__(self, size_average=True):
        super(_Loss, self).__init__()
        self.size_average = size_average


class _WeightedLoss(_Loss):
    def __init__(self, weight=None, size_average=True):
        super(_WeightedLoss, self).__init__(size_average)
        self.register_buffer('weight', weight)


class P3SGD(optim.Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = P3SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of P3SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.
        Let p, g, v, :math:`\rho`, and d denote the parameters, gradient,
        velocity, momentum, and dampening respectively.

        Considering the specific case of Momentum, the update can be written as

        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
             v = \rho * v + lr * g \\
             p = p - v

        Momentum with dampening is modified as

        .. math::

             v = \rho * v + (1 - d) * g \\
             p = p - lr * v

        Finally, the Nesterov momentum version can be analogously modified,
        and in this case the dampening term must always be 0.

    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(P3SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(P3SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is not None:

                    p_data = p.grad.data
                    if weight_decay != 0:
                        temp = torch.mul(p.data, weight_decay)
                        p_data = torch.add(p_data, temp)

                    if momentum != 0:
                        param_state = self.state[p]
                        if 'momentum_buffer' not in param_state:
                            zeros = torch.zeros_like(p.data)
                            mb = zeros
                            param_state['momentum_buffer'] = zeros
                            mb.mul_(momentum).add_(p_data)
                        else:
                            mb = param_state['momentum_buffer']
                            mb.mul_(momentum)
                            mb.add_(1 - dampening, p_data)
                        if nesterov:
                            temp = torch.add(mb, momentum)
                            p_data = torch.add(p_data, temp)
                        else:
                            p_data = mb

            p.data.add_(-group['lr'], p_data)

        return loss


class P3Dropout(nn.Module):
    r"""During training, randomly zeroes some of the elements of the input
    tensor with probability *p* using samples from a bernoulli distribution.
    The elements to zero are randomized on every forward call.
    This has proven to be an effective technique for regularization and
    preventing the co-adaptation of neurons as described in the paper
    `Improving neural networks by preventing co-adaptation of feature
    detectors`_ .
    Furthermore, the outputs are scaled by a factor of :math:`\frac{1}{1-p}` during
    training. This means that during evaluation the module simply computes an
    identity function.
    Args:
        p: probability of an element to be zeroed. Default: 0.5
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``
    Shape:
        - Input: `Any`. Input can be of any shape
        - Output: `Same`. Output is of the same shape as input
    Examples::
        >>> m = nn.Dropout(p=0.2)
        >>> input = torch.randn(20, 16)
        >>> output = m(input)
    .. _Improving neural networks by preventing co-adaptation of feature
        detectors: https://arxiv.org/abs/1207.0580
    """

    def __init__(self, p=0.5, inplace=False):
        super(P3Dropout, self).__init__()
        # TODO Implement me
        # Check if p is within bounds
        if (p < 0 or p > 1):
            sys.exit("p is out of bounds. Must be between 0 and 1.")

        self.p = p
        self.inplace = inplace

    def forward(self, input):
        if  (self.p == 0):
            return input
        input = input.data
        mask_array = np.random.binomial(1, self.p, size = input.size())

        return Variable(torch.from_numpy(input.numpy() * mask_array * (1/(1-self.p))).float())


    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return (self.__class__.__name__ + '(' + 'p=' + str(self.p) +
                inplace_str + ')')


class P3Dropout2d(nn.Module):
    r"""Randomly zeroes whole channels of the input tensor.
    The channels to zero-out are randomized on every forward call.
    Usually the input comes from :class:`nn.Conv2d` modules.
    As described in the paper
    `Efficient Object Localization Using Convolutional Networks`_ ,
    if adjacent pixels within feature maps are strongly correlated
    (as is normally the case in early convolution layers) then i.i.d. dropout
    will not regularize the activations and will otherwise just result
    in an effective learning rate decrease.
    In this case, :func:`nn.Dropout2d` will help promote independence between
    feature maps and should be used instead.
    Args:
        p (float, optional): probability of an element to be zero-ed.
        inplace (bool, optional): If set to ``True``, will do this operation
            in-place
    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)
    Examples::
        >>> m = nn.Dropout2d(p=0.2)
        >>> input = torch.randn(20, 16, 32, 32)
        >>> output = m(input)
    .. _Efficient Object Localization Using Convolutional Networks:
       http://arxiv.org/abs/1411.4280
    """

    def __init__(self, p=0.5, inplace=False):
        super(P3Dropout2d, self).__init__()
        if p < 0 and p > 1:
            raise ValueError("dropout probability has to be between 0 and 1")
        self.p = p
        self.inplace = inplace

    def forward(self, input):
        if  (self.p == 0):
            return input
        input = input.data
        mask_array = np.random.binomial(1, 1 - self.p, size = (input.shape[0], input.shape[1]))
        temp = np.zeros((input.shape[0], input.shape[1], input.shape[2], input.shape[3]))
        temp[(mask_array>0),:,:] = np.ones((input.shape[2], input.shape[3]))
        return Variable(torch.from_numpy(input.numpy() * temp).float())

    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return self.__class__.__name__ + '(' + 'p=' + str(self.p) + inplace_str + ')'


def linear(input, weight, bias=None):
    """
    Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.
    Shape:
        - Input: :math:`(N, *, in\_features)` where `*` means any number of
          additional dimensions
        - Weight: :math:`(out\_features, in\_features)`
        - Bias: :math:`(out\_features)`
        - Output: :math:`(N, *, out\_features)`
    """
    result = input.matmul(weight.t())
    if bias is not None:
        return result + bias
    else:
        return result


class P3LinearFunction(torch.autograd.Function):
    """See P3Linear for details.
    """

    def forward(self, input, weight, bias=None):
        self.save_for_backward(input, weight, bias)
        return linear(input, weight, bias)

    def backward(self, grad_output):
        input, weight, bias = self.saved_tensors

        grad_input = None
        grad_weight = None
        grad_bias = None

        grad_input = grad_output.matmul(weight)
        grad_weight = grad_output.t().matmul(input)
        grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias


class P3Linear(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = Ax + b`
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``
    Shape:
        - Input: :math:`(N, *, in\_features)` where :math:`*` means any number of
          additional dimensions
        - Output: :math:`(N, *, out\_features)` where all but the last dimension
          are the same shape as the input.
    Attributes:
        weight: the learnable weights of the module of shape
            `(out_features x in_features)`
        bias:   the learnable bias of the module of shape `(out_features)`
    Examples::
        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
    """

    def __init__(self, in_features, out_features, bias=True):
        super(P3Linear, self).__init__()
        # TODO Implement me
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if (bias == True):
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        model = P3LinearFunction()
        return model(input, self.weight, self.bias)

    def __repr__(self):
        return (self.__class__.__name__ + '(' +
                'in_features=' + str(self.in_features) +
                ', out_features=' + str(self.out_features) +
                ', bias=' + str(self.bias is not None) + ')')


def p3relu(input, inplace=False):
    r"""relu(input, inplace=False) -> Tensor
    Applies the rectified linear unit function element-wise.
    :math:`\text{ReLU}(x)= \max(0, x)`
    .. image:: _static/img/activation/ReLU.png
    Args:
        input: torch.Tensor
        inplace: can optionally do the operation in-place. Default: ``False``
    """

    if inplace:
        input.mul_((input > 0).type(type(input.data))).abs_()
        return None
    else:
        return input.clamp(min=0)

class P3ReLU(nn.Module):
    r"""Applies the rectified linear unit function element-wise
    :math:`\text{ReLU}(x)= \max(0, x)`
    .. image:: _static/img/activation/ReLU.png
    Args:
        inplace: can optionally do the operation in-place. Default: ``False``
    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    Examples::
        >>> m = nn.ReLU()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def __init__(self, inplace=False):
        super(P3ReLU, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        return p3relu(input, self.inplace)
        
    def __repr__(self):
        inplace_str = 'inplace' if self.inplace else ''
        return self.__class__.__name__ + '(' \
            + inplace_str + ')'


class P3ELUFunction(torch.autograd.Function):
    r"""Applies element-wise,
    :math:`\text{ELU}(x) = \max(0,x) + \min(0, \alpha * (\exp(x) - 1))`
    Args:
        alpha: the :math:`\alpha` value for the ELU formulation. Default: 1.0
        inplace: can optionally do the operation in-place. Default: ``False``
    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    .. image:: _static/img/activation/ELU.png
    Examples::
        >>> m = nn.ELU()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    def forward(self, input, alpha=torch.Tensor([1]), inplace=torch.Tensor([False])):
        self.save_for_backward(input, alpha, inplace)
        if inplace[0] == 1:
            min_part = (input.exp().add(-1).mul(alpha[0])) * (input.exp().add(-1).mul(alpha[0]) < 0).type(type(input))
            input.mul_((input > 0).type(type(input))).abs_()
            input.add_(min_part)
            return None
        else:
            max_part = input.clamp(min=0)
            min_part = (input.exp().add(-1).mul(alpha[0])) * (input.exp().add(-1).mul(alpha[0]) < 0).type(type(input))
            return max_part.add(min_part)

    def backward(self, grad_output):
        input, alpha, inplace = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0].mul_(((input.exp().mul(alpha)))[input < 0])
        return grad_input

class P3ELU(nn.Module):
    r"""Applies element-wise,
    :math:`\text{ELU}(x) = \max(0,x) + \min(0, \alpha * (\exp(x) - 1))`
    Args:
        alpha: the :math:`\alpha` value for the ELU formulation. Default: 1.0
        inplace: can optionally do the operation in-place. Default: ``False``
    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    .. image:: _static/img/activation/ELU.png
    Examples::
        >>> m = nn.ELU()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def __init__(self, alpha=1., inplace=False):
        super(P3ELU, self).__init__()
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, input):
        model = P3ELUFunction()
        if self.inplace:
            model(input, Variable(torch.Tensor([self.alpha])), Variable(torch.Tensor([self.inplace])))
        else:
            return model(input, Variable(torch.Tensor([self.alpha])), Variable(torch.Tensor([self.inplace])))

    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return (self.__class__.__name__ + '(' +
                'alpha=' + str(self.alpha) + inplace_str + ')')


class P3BCELoss(_WeightedLoss):
    r"""Creates a criterion that measures the Binary Cross Entropy
    between the target and the output:
    The loss can be described as:
    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = - w_n \left[ y_n \cdot \log x_n + (1 - y_n) \cdot \log (1 - x_n) \right],
    where :math:`N` is the batch size. If reduce is ``True``, then
    .. math::
        \ell(x, y) = \begin{cases}
            \operatorname{mean}(L), & \text{if}\; \text{size_average} = \text{True},\\
            \operatorname{sum}(L),  & \text{if}\; \text{size_average} = \text{False}.
        \end{cases}
    This is used for measuring the error of a reconstruction in for example
    an auto-encoder. Note that the targets `y` should be numbers
    between 0 and 1.
    Args:
        weight (Tensor, optional): a manual rescaling weight given to the loss
            of each batch element. If given, has to be a Tensor of size
            "nbatch".
        size_average (bool, optional): By default, the losses are averaged
            over observations for each minibatch. However, if the field
            size_average is set to ``False``, the losses are instead summed for
            each minibatch. Default: ``True``
        reduce (bool, optional): By default, the losses are averaged or summed over
            observations for each minibatch depending on size_average. When reduce
            is False, returns a loss per input/target element instead and ignores
            size_average. Default: True
    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Target: :math:`(N, *)`, same shape as the input
        - Output: scalar. If `reduce` is False, then `(N, *)`, same shape as
          input.
    Examples::
        >>> m = nn.Sigmoid()
        >>> loss = nn.BCELoss()
        >>> input = torch.randn(3, requires_grad=True)
        >>> target = torch.FloatTensor(3).random_(2)
        >>> output = loss(m(input), target)
        >>> output.backward()
    """
    def __init__(self, weight=None, size_average=True, reduce=True):
        super(P3BCELoss, self).__init__(weight, size_average)
        self.reduce = reduce

    def forward(self, input, target):
        _assert_no_grad(target)
        # TODO implement me
        # Convert to numpy for calculation
        

# Define the neural network classes


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = P3Linear(320, 50)
        self.fc2 = P3Linear(50, 10)

    def forward(self, x):
        # F is just a functional wrapper for modules from the nn package
        # see http://pytorch.org/docs/_modules/torch/nn/functional.html
        x = p3relu(F.max_pool2d(self.conv1(x), 2))
        x = p3relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = p3relu(self.fc1(x))
        m_dropout = P3Dropout()
        x = F.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class NewNet(nn.Module):
    def __init__(self):
        super(NewNet, self).__init__()
        self.bn_start = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=4)
        self.conv1_do = P3Dropout2d(p=0.1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=4)
        self.conv2_do = P3Dropout2d(p=0.2)
        self.fc1 = P3Linear(64 * 22 * 22, 250)
        self.fc1_do = nn.Dropout(p=0.3)
        self.fc2 = P3Linear(250, 64)
        self.bn_end = nn.BatchNorm2d(64)
        self.fc_end = P3Linear(64, 10)

    def forward(self, x):
        # F is just a functional wrapper for modules from the nn package
        # see http://pytorch.org/docs/_modules/torch/nn/functional.html
        x = self.bn_start(x)
        x = self.conv1_do(F.max_pool2d(F.relu(self.conv1(x)), 1))
        x = self.conv2_do(F.max_pool2d(F.relu(self.conv2(x)), 1))
        x = x.view(-1, 64 * 22 * 22)
        x = self.fc1_do(F.relu(self.fc1(x)))
        x = F.relu(self.fc2(x))
        x = self.bn_end(x)
        x = self.fc_end(x)

        return F.log_softmax(x, dim=1)

def chooseModel(model_name='default', cuda=True):
    # TODO add all the other models here if their parameter is specified
    if model_name == 'default' or model_name == 'P2Q7DefaultChannelsNet':
        model = Net()
    elif model_name in globals():
        model = globals()[model_name]()
    else:
        raise ValueError('Unknown model type: ' + model_name)

    if args.cuda:
        model.cuda()
    return model


def chooseOptimizer(model, optimizer='sgd'):
    if optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    elif optimizer == 'adam':
        optimizer = optim.Adam(model.parameters())
    elif optimizer == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters())
    else:
        raise ValueError('Unsupported optimizer: ' + args.optimizer)
    return optimizer


def train(model, optimizer, train_loader, tensorboard_writer, callbacklist, epoch, total_minibatch_count):
    # Training
    model.train()
    correct_count = np.array(0)
    for batch_idx, (data, target) in enumerate(train_loader):
        callbacklist.on_batch_begin(batch_idx)
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()

        # Forward prediction step
        output = model(data)
        loss = F.nll_loss(output, target)

        # Backpropagation step
        loss.backward()
        optimizer.step()

        # The batch has ended, determine the
        # accuracy of the predicted outputs
        _, argmax = torch.max(output, 1)

        # target labels and predictions are
        # categorical values from 0 to 9.
        accuracy = (target == argmax.squeeze()).float().mean()
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct_count += pred.eq(target.data.view_as(pred)).cpu().sum()

        batch_logs = {
            'loss': np.array(loss.data[0]),
            'acc': np.array(accuracy.data[0]),
            'size': np.array(len(target))
        }

        batch_logs['batch'] = np.array(batch_idx)
        callbacklist.on_batch_end(batch_idx, batch_logs)

        if args.log_interval != 0 and total_minibatch_count % args.log_interval == 0:
            # put all the logs in tensorboard
            for name, value in six.iteritems(batch_logs):
                tensorboard_writer.add_scalar(name, value, global_step=total_minibatch_count)

            # put all the parameters in tensorboard histograms
            for name, param in model.named_parameters():
                name = name.replace('.', '/')
                tensorboard_writer.add_histogram(name, param.data.cpu().numpy(), global_step=total_minibatch_count)
                # tensorboard_writer.add_histogram(name + '/gradient', param.grad.data.cpu().numpy(), global_step=total_minibatch_count)

        total_minibatch_count += 1

    # display the last batch of images in tensorboard
    img = torchvision.utils.make_grid(255 - data.data, normalize=True, scale_each=True)
    tensorboard_writer.add_image('images', img, global_step=total_minibatch_count)

    return total_minibatch_count


def test(model, test_loader, tensorboard_writer, callbacklist, epoch, total_minibatch_count):
    # Validation Testing
    model.eval()
    test_loss = 0
    correct = 0
    progress_bar = tqdm(test_loader, desc='Validation')
    for data, target in progress_bar:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0]  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_size = np.array(len(test_loader.dataset), np.float32)
    test_loss /= test_size

    acc = np.array(correct, np.float32) / test_size
    epoch_logs = {'val_loss': np.array(test_loss),
                  'val_acc': np.array(acc)}
    for name, value in six.iteritems(epoch_logs):
        tensorboard_writer.add_scalar(name, value, global_step=total_minibatch_count)
    callbacklist.on_epoch_end(epoch, epoch_logs)
    progress_bar.write(
        'Epoch: {} - validation test results - Average val_loss: {:.4f}, val_acc: {}/{} ({:.2f}%)'.format(
            epoch, test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    return acc


def run_experiment(args):
    total_minibatch_count = 0
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    epochs_to_run = args.epochs
    tensorboard_writer, callbacklist, train_loader, test_loader = prepareDatasetAndLogging(args)
    model = chooseModel(args.model)
    # tensorboard_writer.add_graph(model, images[:2])
    optimizer = chooseOptimizer(model, args.optimizer)
    # Run the primary training loop, starting with validation accuracy of 0
    val_acc = 0
    callbacklist.on_train_begin()
    for epoch in range(1, epochs_to_run + 1):
        callbacklist.on_epoch_begin(epoch)
        # train for 1 epoch
        total_minibatch_count = train(model, optimizer, train_loader, tensorboard_writer,
                                      callbacklist, epoch, total_minibatch_count)
        # validate progress on test dataset
        val_acc = test(model, test_loader, tensorboard_writer,
                       callbacklist, epoch, total_minibatch_count)
    callbacklist.on_train_end()
    tensorboard_writer.close()

    if args.dataset == 'fashion_mnist' and val_acc > 0.8 and val_acc <= 1.0:
        print("Congratulations, you beat the Question 9 minimum of 0.80 with ({:.2f}%) validation accuracy!".format(val_acc))


if __name__ == '__main__':
    args = parser.parse_args()
    # Run the primary training and validation loop
    run_experiment(args)
