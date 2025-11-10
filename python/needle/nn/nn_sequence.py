"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        # 1 / (1 + exp(-x))
        return ops.power_scalar(ops.add_scalar(ops.exp(ops.negate(x)), 1.0), -1)

class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        assert nonlinearity in ['tanh', 'relu']
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity
        k = 1.0 / hidden_size
        bound = k ** 0.5
        # Parameters
        self.W_ih = Parameter(init.rand(input_size, hidden_size, low=-bound, high=bound, device=device, dtype=dtype, requires_grad=True))
        self.W_hh = Parameter(init.rand(hidden_size, hidden_size, low=-bound, high=bound, device=device, dtype=dtype, requires_grad=True))
        if bias:
            self.bias_ih = Parameter(init.rand(hidden_size, low=-bound, high=bound, device=device, dtype=dtype, requires_grad=True))
            self.bias_hh = Parameter(init.rand(hidden_size, low=-bound, high=bound, device=device, dtype=dtype, requires_grad=True))
        else:
            self.bias_ih = None
            self.bias_hh = None

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        # Ensure inputs and hidden state live on same device as parameters to avoid cross-device matmul errors
        param_device = self.W_ih.device
        assert X.device == param_device, f"RNNCell input device {X.device} != param device {param_device}"
        bs = X.shape[0]
        if h is None:
            h = init.zeros(bs, self.hidden_size, device=param_device, dtype=X.dtype, requires_grad=False)
        out = ops.matmul(X, self.W_ih) + ops.matmul(h, self.W_hh)
        if self.bias_ih is not None:
            b_ih = ops.reshape(self.bias_ih, (1, self.hidden_size))
            b_hh = ops.reshape(self.bias_hh, (1, self.hidden_size))
            out = out + ops.broadcast_to(b_ih, out.shape) + ops.broadcast_to(b_hh, out.shape)
        out = ops.tanh(out) if self.nonlinearity == 'tanh' else ops.relu(out)
        return out


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.bias = bias
        self.rnn_cells: List[RNNCell] = []
        # First layer takes input_size, subsequent take hidden_size
        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_size
            cell = RNNCell(in_size, hidden_size, bias=bias, nonlinearity=nonlinearity, device=device, dtype=dtype)
            self.rnn_cells.append(cell)

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        # X: (seq_len, bs, input_size); h0: (num_layers, bs, hidden_size)
        # Align input sequence device with parameters (first cell)
        param_device = self.rnn_cells[0].W_ih.device
        assert X.device == param_device, f"RNN input device {X.device} != param device {param_device}"
        seq_len, bs, _ = X.shape
        time_steps = ops.split(X, axis=0)
        if h0 is None:
            h_prev = [init.zeros(bs, self.hidden_size, device=X.device, dtype=X.dtype, requires_grad=False)
                      for _ in range(self.num_layers)]
        else:
            assert h0.device == param_device, f"RNN h0 device {h0.device} != param device {param_device}"
            h_prev = list(ops.split(h0, axis=0))
        outputs = []
        for x_t in time_steps:
            layer_input = x_t
            new_states = []
            for layer_idx, cell in enumerate(self.rnn_cells):
                h_out = cell(layer_input, h_prev[layer_idx])
                layer_input = h_out
                new_states.append(h_out)
            h_prev = new_states
            outputs.append(layer_input)
        output = ops.stack(outputs, axis=0)
        h_n = ops.stack(h_prev, axis=0)
        return output, h_n


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        k = 1.0 / hidden_size
        bound = k ** 0.5
        
        self.W_ih = Parameter(init.rand(input_size, 4 * hidden_size, low=-bound, high=bound, device=device, dtype=dtype, requires_grad=True))
        self.W_hh = Parameter(init.rand(hidden_size, 4 * hidden_size, low=-bound, high=bound, device=device, dtype=dtype, requires_grad=True))
        
        if bias:
            self.bias_ih = Parameter(init.rand(4 * hidden_size, low=-bound, high=bound, device=device, dtype=dtype, requires_grad=True))
            self.bias_hh = Parameter(init.rand(4 * hidden_size, low=-bound, high=bound, device=device, dtype=dtype, requires_grad=True))
        else:
            self.bias_ih = None
            self.bias_hh = None
            
        # Create sigmoid activation
        self.sigmoid = Sigmoid()

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        # Ensure inputs and hidden state live on same device as parameters
        param_device = self.W_ih.device
        assert X.device == param_device, f"LSTMCell input device {X.device} != param device {param_device}"
        bs = X.shape[0]
        
        # Initialize hidden and cell states if not provided
        if h is None:
            h0 = init.zeros(bs, self.hidden_size, device=param_device, dtype=X.dtype, requires_grad=False)
            c0 = init.zeros(bs, self.hidden_size, device=param_device, dtype=X.dtype, requires_grad=False)
        else:
            h0, c0 = h
        
        # Compute the combined gate activations
        gi = ops.matmul(X, self.W_ih) + ops.matmul(h0, self.W_hh)
        
        # Add biases if they exist
        if self.bias_ih is not None:
            gi = gi + ops.broadcast_to(ops.reshape(self.bias_ih, (1, 4 * self.hidden_size)), gi.shape)
            gi = gi + ops.broadcast_to(ops.reshape(self.bias_hh, (1, 4 * self.hidden_size)), gi.shape)
        
        # Split gi into gates
        gi_reshaped = ops.reshape(gi, (bs, 4, self.hidden_size))
        gate_list = ops.split(gi_reshaped, 1)
        
        i_gate = gate_list[0]
        f_gate = gate_list[1]
        g_gate = gate_list[2] 
        o_gate = gate_list[3]
        
        i = self.sigmoid(i_gate)
        f = self.sigmoid(f_gate)
        g = ops.tanh(g_gate)
        o = self.sigmoid(o_gate)
        
        # Update cell and hidden states
        c_new = f * c0 + i * g
        h_new = o * ops.tanh(c_new)
        
        return h_new, c_new



class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.lstm_cells: List[LSTMCell] = []
        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_size
            cell = LSTMCell(in_size, hidden_size, bias=bias, device=device, dtype=dtype)
            self.lstm_cells.append(cell)

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        param_device = self.lstm_cells[0].W_ih.device
        assert X.device == param_device, f"LSTM input device {X.device} != param device {param_device}"
        seq_len, bs, _ = X.shape
        time_steps = ops.split(X, axis=0)
        if h is None:
            h_prev = [init.zeros(bs, self.hidden_size, device=X.device, dtype=X.dtype, requires_grad=False)
                      for _ in range(self.num_layers)]
            c_prev = [init.zeros(bs, self.hidden_size, device=X.device, dtype=X.dtype, requires_grad=False)
                      for _ in range(self.num_layers)]
        else:
            h0, c0 = h
            assert h0.device == param_device and c0.device == param_device, (
                f"LSTM state devices {(h0.device, c0.device)} != param device {param_device}"
            )
            h_prev = list(ops.split(h0, axis=0))
            c_prev = list(ops.split(c0, axis=0))
        outputs = []
        for x_t in time_steps:
            layer_input = x_t
            new_h = []
            new_c = []
            for layer_idx, cell in enumerate(self.lstm_cells):
                h_out, c_out = cell(layer_input, (h_prev[layer_idx], c_prev[layer_idx]))
                layer_input = h_out
                new_h.append(h_out)
                new_c.append(c_out)
            h_prev, c_prev = new_h, new_c
            outputs.append(layer_input)
        output = ops.stack(outputs, axis=0)
        h_n = ops.stack(h_prev, axis=0)
        c_n = ops.stack(c_prev, axis=0)
        return output, (h_n, c_n)

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(init.randn(num_embeddings, embedding_dim, mean=0.0, std=1.0, device=device, dtype=dtype, requires_grad=True))

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        # Align indices with embedding weight device
        assert x.device == self.weight.device, f"Embedding input device {x.device} != weight device {self.weight.device}"
        seq_len, bs = x.shape
        time_indices = ops.split(x, axis=0)  # tuple length seq_len, each (bs,)
        embeddings = []
        for idx in time_indices:
            oh = init.one_hot(self.num_embeddings, idx, device=idx.device, dtype="float32")
            emb = ops.matmul(oh, self.weight)
            embeddings.append(emb)
        return ops.stack(embeddings, axis=0)