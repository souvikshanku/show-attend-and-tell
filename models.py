import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, device):
        super(Encoder, self).__init__()
        self.device = device

        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, img):
        img = F.relu(self.conv1(img))
        img = self.pool(img)
        img = F.relu(self.conv2(img))
        img = self.pool(img)
        img = F.relu(self.conv3(img))
        img = self.pool(img)
        return img


class Attention(nn.Module):
    def __init__(self, enc_dim, device):
        super(Attention, self).__init__()
        self.device = device

        # TODO: Make these dynamic
        self.W = nn.Linear(32, 128)
        self.U = nn.Linear(enc_dim, 128)
        self.v = nn.Linear(128, 1)
        self.fc_beta = nn.Linear(enc_dim, 32)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img_features, prev_hidden):
        batch_size = img_features.shape[0]
        num_kernels = img_features.shape[1]
        img_features = img_features.view(batch_size, num_kernels, -1)
        img_parts = img_features.permute(0, 2, 1)

        W_s = self.W(img_parts)
        U_h = self.U(prev_hidden).unsqueeze(1)
        att = self.tanh(W_s + U_h)
        score = self.v(att).view(batch_size, -1)
        att_weights = self.softmax(score)
        context = (img_parts * att_weights.unsqueeze(2)).sum(1)

        beta = self.sigmoid(self.fc_beta(prev_hidden))
        context *= beta

        return context, att_weights


class Decoder(nn.Module):
    def __init__(self, enc_dim, device):
        super(Decoder, self).__init__()
        self.device = device
        self.enc_dim = enc_dim
        # self.context_size = 32

        self.encoder = Encoder(self.device)
        self.attention = Attention(self.enc_dim, self.device)
        self.rnn = nn.LSTMCell(input_size=12 + 32, hidden_size=self.enc_dim)
        self.init_h = nn.Linear(64, self.enc_dim)
        self.init_c = nn.Linear(64, self.enc_dim)
        self.out_fc = nn.Linear(self.enc_dim, 11)

    def init_hiddens(self, img_features):
        batch_size = img_features.shape[0]
        num_kernels = img_features.shape[1]
        img_features = img_features.view(batch_size, num_kernels, -1)
        avg = img_features.mean(dim=1)

        h_t = self.init_h(avg)
        h_t = F.tanh(h_t)
        c_t = self.init_c(avg)
        c_t = F.tanh(c_t)

        return h_t, c_t

    def forward(self, img, label):
        img_features = self.encoder(img)
        label = label.permute(1, 0, 2)

        h_t, c_t = self.init_hiddens(img_features)

        output = []
        alphas = []

        for t in range(len(label) - 1):
            context, alpha = self.attention(img_features, h_t)

            inp = torch.concat([label[t + 1], context], dim=1)
            h_t, c_t = self.rnn(inp, (h_t, c_t))

            out = self.out_fc(h_t)

            output.append(out)
            alphas.append(alpha)

        output = torch.stack(output)
        output = F.log_softmax(output, dim=-1)
        output = output.permute(1, 0, 2)

        return output
