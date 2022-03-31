import torch
import torch.nn as nn
import torch.nn.utils as utils

# (B, C, T) -> (B, C, T-s)
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]  

class SqueezeChannels(nn.Module):
    def __init__(self):
        super(SqueezeChannels, self).__init__()

    def forward(self, x):
        return x.squeeze(2)


class FCN(nn.Module):
    def __init__(self, num_classes):
        super(FCN, self).__init__()

        self.num_classes = num_classes
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=8, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, padding='same'),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.network = nn.Sequential(
            self.conv_block1,
            self.conv_block2,
            self.conv_block3,
            nn.AdaptiveAvgPool1d(1),
            SqueezeChannels(),
        
        )
    
    def forward(self, x):
        return self.network(x)



class DilatedBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False):
        super(DilatedBlock, self).__init__()
        
        padding = (kernel_size - 1) * dilation

        self.conv_block1 = nn.Sequential(
            utils.weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size, 
            padding=padding, dilation=dilation)), 
            Chomp1d(padding),
            nn.LeakyReLU()
        )

        self.conv_block2 = nn.Sequential(
            utils.weight_norm(nn.Conv1d(out_channels, out_channels, kernel_size, 
            padding=padding, dilation=dilation)), 
            Chomp1d(padding),
            nn.LeakyReLU()
        )

        # whether apply residual connection
        self.upordownsample = torch.nn.Conv1d(
            in_channels, out_channels, 1
        ) if in_channels != out_channels else None

        self.relu = torch.nn.LeakyReLU() if final else None


    def forward(self, x):
        out = self.conv_block1(x)
        out = self.conv_block2(out)

        res = x if self.upordownsample is None else self.upordownsample(x)

        if self.relu is None:
            return out + res
        else:
            return self.relu(out+res)
  


class DilatedConvolution(nn.Module):
    def __init__(self, in_channels, embedding_channels, out_channels, depth, reduced_size, kernel_size, num_classes) -> None:
        super(DilatedConvolution, self).__init__()
        
        layers = []
        # dilation size will be doubled at each step according to TLoss
        dilation_size = 1

        for i in range(depth):
            block_in_channels = in_channels if i == 0 else embedding_channels
            layers += [DilatedBlock(block_in_channels, embedding_channels, kernel_size, dilation_size)]
            dilation_size *= 2
        
        layers += [DilatedBlock(embedding_channels, reduced_size, kernel_size, dilation_size, final=True)]

        self.global_average_pool = nn.AdaptiveAvgPool1d(1)
        
        # 注意， dilated中用的是global max pool
        self.network = nn.Sequential(*layers,
                                     nn.AdaptiveMaxPool1d(1),
                                     SqueezeChannels(),
                                     nn.Linear(reduced_size, out_channels),
                                     )

    def forward(self, x):
        return self.network(x)

class Classifier(nn.Module):
    def __init__(self, input_dims, output_dims) -> None:
        super(Classifier, self).__init__()

        self.dense = nn.Linear(input_dims, output_dims)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        
        return self.softmax(self.dense(x))

class NonLinearClassifier(nn.Module):
    def __init__(self, input_dim, embedding_dim, output_dim, dropout=0.2) -> None:
        super(NonLinearClassifier, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, output_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.net(x)

# for single step
class RNNDecoder(nn.Module):
    def __init__(self, input_dim=1, embedding_dim=320) -> None:
        super(RNNDecoder, self).__init__()
        self.grucell1 = nn.GRUCell(input_size=input_dim, hidden_size=embedding_dim)
        self.grucell2 = nn.GRUCell(input_size=embedding_dim, hidden_size=embedding_dim)
        self.grucell3 = nn.GRUCell(input_size=embedding_dim, hidden_size=embedding_dim)

        self.linear = nn.Linear(in_features=embedding_dim, out_features=1)

    # x : single time step (batch_size, 1)
    # TODO 是否将训练循环改到train.py中
    def forward(self, h1, h2, h3, x):
        hidden1 = self.grucell1(x, h1)
        hidden2 = self.grucell2(hidden1, h2)
        hidden3 = self.grucell3(hidden2, h3)

        out = self.linear(hidden3)

        return hidden1, hidden2, hidden3, out




if __name__ == '__main__':
    pass


# TODO
# add args（depth, in_channels, out_channels, reduced_size, embedding_channels, kernel_size  in train.py
# finish dataloader.py