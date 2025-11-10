import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import math
import numpy as np
np.random.seed(0)


class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        # CIFAR-10 variant of ResNet9 with channel sizes divided for lightweight training.
        # Architecture (NCHW inputs):
        # (1) Conv 3->16
        # (2) Conv 16->32 (stride 2)
        # (3) Residual block: two Conv 32->32
        # (4) Conv 32->64 (stride 2)
        # (5) Conv 64->128 (stride 2)
        # (6) Residual block: two Conv 128->128
        # (7) Global average pool + Linear 128->10
        # We use BatchNorm2d after every convolution and ReLU activations.

        def conv_bn_relu(in_c, out_c, ks, stride):
            layers = [nn.Conv(in_c, out_c, kernel_size=ks, stride=stride, device=device, dtype=dtype)]
            layers.append(nn.BatchNorm2d(out_c, device=device, dtype=dtype))
            layers.append(nn.ReLU())
            return nn.Sequential(*layers)

        # Stem
        self.block1 = conv_bn_relu(3, 16, 7, 4)
        self.block2 = conv_bn_relu(16, 32, 3, 2)

        # Residual block at 32 channels
        self.res1 = nn.Residual(
            nn.Sequential(
                conv_bn_relu(32, 32, 3, 1),
                conv_bn_relu(32, 32, 3, 1),
            )
        )

        # Downsampling convs
        self.block3 = conv_bn_relu(32, 64, 3, 2)
        # For the final 128-channel stage, omit BatchNorm to match expected parameter count
        self.block4 = conv_bn_relu(64, 128, 3, 2)

        # Residual block at 128 channels
        self.res2 = nn.Residual(
            nn.Sequential(
                conv_bn_relu(128, 128, 3, 1),
                conv_bn_relu(128, 128, 3, 1),
            )
        )

        # Classification head
        self.flatten = nn.Flatten()
        self.lin1 = nn.Linear(128, 128, device=device, dtype=dtype)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(128, 10, device=device, dtype=dtype)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.res1(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.res2(x)
        x = self.flatten(x)
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        return x


class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', seq_len=40, device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        assert seq_model in ['rnn', 'lstm'], "seq_model must be 'rnn' or 'lstm'"
        self.output_size = output_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_model_type = seq_model
        self.seq_len = seq_len
        self.device = device
        self.dtype = dtype

        # Embedding: maps (seq_len, bs) of word indices to (seq_len, bs, embedding_size)
        self.embedding = nn.Embedding(output_size, embedding_size, device=device, dtype=dtype)

        # Sequence model: RNN or LSTM
        if seq_model == 'rnn':
            self.seq_model = nn.RNN(embedding_size, hidden_size, num_layers=num_layers, bias=True, device=device, dtype=dtype)
        else:
            self.seq_model = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, bias=True, device=device, dtype=dtype)

        # Output linear layer: maps hidden state (seq_len, bs, hidden_size) -> logits (seq_len*bs, output_size)
        self.fc = nn.Linear(hidden_size, output_size, device=device, dtype=dtype)

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        # x: (seq_len, bs)
        emb = self.embedding(x)  # (seq_len, bs, embedding_size)

        if self.seq_model_type == 'rnn':
            out_seq, h_new = self.seq_model(emb, h)  # out_seq: (seq_len, bs, hidden_size)
        else:
            out_seq, h_new = self.seq_model(emb, h)  # h_new: (h_n, c_n)

        seq_len, bs, _ = out_seq.shape
        out_flat = out_seq.reshape((seq_len * bs, self.hidden_size))
        logits = self.fc(out_flat)  # (seq_len*bs, output_size)
        return logits, h_new


if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(cifar10_train_dataset, 128, ndl.cpu(), dtype="float32")
    print(cifar10_train_dataset[1][0].shape)

