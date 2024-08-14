import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from resources.models.model_clam import Attn_Net, Attn_Net_Gated


class MIL_fc(nn.Module):
    def __init__(self, size_arg = "small", dropout = 0., n_classes = 2, top_k=1,
                 embed_dim=1024):
        super().__init__()
        assert n_classes == 2
        self.size_dict = {"small": [embed_dim, 512]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        self.fc = nn.Sequential(*fc)
        self.classifier=  nn.Linear(size[1], n_classes)
        self.top_k=top_k

    def forward(self, h, return_features=False):
        h = self.fc(h)
        logits  = self.classifier(h) # K x 2
        
        y_probs = F.softmax(logits, dim = 1)
        top_instance_idx = torch.topk(y_probs[:, 1], self.top_k, dim=0)[1].view(1,)
        top_instance = torch.index_select(logits, dim=0, index=top_instance_idx)
        Y_hat = torch.topk(top_instance, 1, dim = 1)[1]
        Y_prob = F.softmax(top_instance, dim = 1) 
        results_dict = {}

        if return_features:
            top_features = torch.index_select(h, dim=0, index=top_instance_idx)
            results_dict.update({'features': top_features})
        return top_instance, Y_prob, Y_hat, y_probs, results_dict


class MIL_fc_mc(nn.Module):
    def __init__(self, size_arg = "small", dropout = 0., n_classes = 2, top_k=1, embed_dim=1024):
        super().__init__()
        assert n_classes > 2
        self.size_dict = {"small": [embed_dim, 512]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        self.fc = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        self.top_k=top_k
        self.n_classes = n_classes
        assert self.top_k == 1
    
    def forward(self, h, return_features=False):       
        h = self.fc(h)
        logits = self.classifiers(h)

        y_probs = F.softmax(logits, dim = 1)
        m = y_probs.view(1, -1).argmax(1)
        top_indices = torch.cat(((m // self.n_classes).view(-1, 1), (m % self.n_classes).view(-1, 1)), dim=1).view(-1, 1)
        top_instance = logits[top_indices[0]]

        Y_hat = top_indices[1]
        Y_prob = y_probs[top_indices[0]]
        
        results_dict = {}

        if return_features:
            top_features = torch.index_select(h, dim=0, index=top_indices[0])
            results_dict.update({'features': top_features})
        return top_instance, Y_prob, Y_hat, y_probs, results_dict


class MIL_fc_reg(nn.Module):
    def __init__(self, size_arg="small", dropout=0., n_classes=1, top_k=1,
                 embed_dim=1024):
        super().__init__()
        assert n_classes == 1
        self.size_dict = {"small": [embed_dim, 512]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        self.fc = nn.Sequential(*fc)
        self.regressor = nn.Linear(size[1], 1)
        self.top_k = top_k

    def forward(self, h, return_features=False): # h dims: batch_size x hidden_dim (1024 default)
        h = self.fc(h) # batch_size x 512
        log_risks = self.regressor(h)  # batch_size x 1 the predicted log (risk) for the patch

        risks = torch.exp(log_risks)
        top_instance_idx = torch.topk(risks[:, 0], self.top_k, dim=0)[1]#.view(1, )
        top_log_risks_select = torch.index_select(log_risks, dim=0, index=top_instance_idx)
        top_log_risks_mean = top_log_risks_select.mean(dim = 0)
        # top_instance = torch.index_select(log_risks, dim=0, index=top_instance_idx)
        # log_risk_hat = torch.topk(top_instance, 1, dim=1)[1]
        top_risk_select_mean = torch.exp(top_log_risks_mean)
        results_dict = {}

        if return_features:
            top_features = torch.index_select(h, dim=0, index=top_instance_idx)
            results_dict.update({'features': top_features})
        return top_log_risks_mean, top_risk_select_mean, results_dict, torch.transpose(log_risks, 1, 0)

class MIL_fc_reg_att(nn.Module):
    def __init__(self, size_arg="small", dropout=0., n_classes=1,
                 embed_dim=1024, gate = True):
        super().__init__()
        assert n_classes == 1
        self.size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.regressor = nn.Linear(size[1], 1)

    def forward(self, h, return_features=False, attention_only = False): # h dims: batch_size x hidden_dim (512 defined in __init__)
        A, h = self.attention_net(h) # h dims: batch_size x 512; A dims: batch_size x 1 since regression (n_class = 1)
        A = torch.transpose(A, 1, 0)  # 1 x batch_size
        if attention_only:
            return A
        # A_raw = A ### find a way to output raw attention map in the future
        A = F.softmax(A, dim=1)  #

        M = torch.mm(A, h) # 1x batch_size matmul batch_size x hidden_dim >> 1 x hidden_dim (512 defined in __init__)
                            # This is the attention weighted average feature vector dim 1 x hidden_dim (512 defined in __init__)

        log_risks = self.regressor(M)  # 1x1

        risks = torch.exp(log_risks)
        results_dict = {}

        if return_features:

            results_dict.update({'features': M})

        return log_risks, risks, results_dict, A


class MIL_fc_reg_top_k_att(nn.Module):
    def __init__(self, size_arg="small", dropout=0., n_classes=1, top_k = 10,
                 embed_dim=1024, gate = True):
        super().__init__()
        assert n_classes == 1
        self.size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.regressor = nn.Linear(size[1], 1)
        self.top_k = top_k

    def forward(self, h, return_features=False, attention_only = False): # h dims: batch_size x hidden_dim (512 defined in __init__)
        A, h = self.attention_net(h) # h dims: batch_size x 512; A dims: batch_size x 1 since regression (n_class = 1)
        log_risks_start = self.regressor(h)  # batch_size x 1 the predicted log (risk) for the patch
        top_instance_idx = torch.topk(log_risks_start[:, 0], self.top_k, dim=0)[1]#.view(1, )
        top_h = torch.index_select(h, dim=0, index=top_instance_idx)
        top_A = torch.index_select(A, dim=0, index=top_instance_idx)

        top_A = torch.transpose(top_A, 1, 0)  # 1 x batch_size
        if attention_only:
            return top_A
        # A_raw = A ### find a way to output raw attention map in the future
        top_A = F.softmax(top_A, dim=1)  #

        M = torch.mm(top_A, top_h) # 1x batch_size matmul batch_size x hidden_dim >> 1 x hidden_dim (512 defined in __init__)
                            # This is the attention weighted average feature vector dim 1 x hidden_dim (512 defined in __init__)

        log_risks = self.regressor(M)  # 1x1

        risks = torch.exp(log_risks)
        results_dict = {}

        if return_features:

            results_dict.update({'features': M})

        return log_risks, risks, results_dict, top_A


"""from https://github.com/tueimage/MAD-MIL/blob/main/models/model_abmil.py, MAD-MIL
modified to fit the regression task. Original class name is ABMIL_Multihead
"""


class MAD_MIL_reg(nn.Module):
    def __init__(self, gate=True, size_arg="small", dropout=0.25, n_classes=1, n_heads=2, head_size="small", embed_dim=1024):
        super(MAD_MIL_reg, self).__init__()
        assert n_classes == 1
        self.n_heads = n_heads
        self.size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384], "tiny": [embed_dim, 128, 16]}
        self.size = self.size_dict[size_arg]

        if self.size[1] % self.n_heads != 0:
            print("The feature dim should be divisible by num_heads!! Do't worry, we will fix it for you.")
            self.size[1] = math.ceil(self.size[1] / self.n_heads) * self.n_heads

        self.step = self.size[1] // self.n_heads

        if head_size == "tiny":
            self.dim = self.step // 4
        elif head_size == "small":
            self.dim = self.step // 2
        elif head_size == "big":
            self.dim = self.size[2]
        else:
            self.dim = self.step


        fc = [nn.Linear(self.size[0], self.size[1]), nn.ReLU(), nn.Dropout(dropout)]

        if gate:
            att_net = [Attn_Net_Gated(L=self.step, D=self.dim, dropout=dropout, n_classes=1) for _ in
                       range(self.n_heads)]
        else:
            att_net = [Attn_Net(L=self.step, D=self.dim, dropout=dropout, n_classes=1) for _ in range(self.n_heads)]

        self.net_general = nn.Sequential(*fc)
        self.attention_net = nn.ModuleList(att_net)
        self.regressor = nn.Linear(self.size[1], 1)
        # self.n_classes = n_classes
        # initialize_weights(self)

    # def relocate(self):
    #     """
    #     Relocates the model to GPU if available, else to CPU.
    #     """
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     self.net_general = self.net_general.to(device)
    #     self.attention_net = self.attention_net.to(device)
    #     self.classifiers = self.classifiers.to(device)

    def forward(self, h, return_features=False, attention_only=False):
        """
        Forward pass of the model.

        Args:
            h (torch.Tensor): Input tensor
            attention_only (bool): Whether to return only attention weights
            return_features (bool): whether to return attention matrix M

        Returns:
            tuple: Tuple containing logits, predicted probabilities, predicted labels, attention weights, and attention weights before softmax

        """
        device = h.device

        h = self.net_general(h)
        N, C = h.shape

        # Multihead Input
        h = h.reshape(N, self.n_heads, C // self.n_heads)

        A = torch.empty(N, self.n_heads, 1).float().to(device)
        for nn in range(self.n_heads):
            a, _ = self.attention_net[nn](h[:, nn, :])
            A[:, nn, :] = a

        A = torch.transpose(A, 2, 0)  # KxheadsxN
        if attention_only:
            return A
        A_raw = A

        A = F.softmax(A, dim=-1)  # softmax over N

        # Multihead Output
        M = torch.empty(1, self.size[1]).float().to(device)
        for nn in range(self.n_heads):
            m = torch.mm(A[:, nn, :], h[:, nn, :])
            M[:, self.step * nn: self.step * nn + self.step] = m

        # Singlehead Classification
        log_risks = self.regressor(M)
        risks = torch.exp(log_risks)
        results_dict = {}

        if return_features:

            results_dict.update({'features': M})

        return log_risks, risks, results_dict, A


class MAD_MIL(nn.Module):
    def __init__(self, gate=True, size_arg="small", dropout=0.25, n_classes=1, n_heads=2, head_size="small", embed_dim=1024):
        super(MAD_MIL, self).__init__()

        self.n_heads = n_heads
        self.size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384], "tiny": [embed_dim, 128, 16]}
        self.size = self.size_dict[size_arg]

        if self.size[1] % self.n_heads != 0:
            print("The feature dim should be divisible by num_heads!! Do't worry, we will fix it for you.")
            self.size[1] = math.ceil(self.size[1] / self.n_heads) * self.n_heads

        self.step = self.size[1] // self.n_heads

        if head_size == "tiny":
            self.dim = self.step // 4
        elif head_size == "small":
            self.dim = self.step // 2
        elif head_size == "big":
            self.dim = self.size[2]
        else:
            self.dim = self.step


        fc = [nn.Linear(self.size[0], self.size[1]), nn.ReLU(), nn.Dropout(dropout)]

        if gate:
            att_net = [Attn_Net_Gated(L=self.step, D=self.dim, dropout=dropout, n_classes=1) for _ in
                       range(self.n_heads)]
        else:
            att_net = [Attn_Net(L=self.step, D=self.dim, dropout=dropout, n_classes=1) for _ in range(self.n_heads)]

        self.net_general = nn.Sequential(*fc)
        self.attention_net = nn.ModuleList(att_net)
        self.classifiers = nn.Linear(self.size[1], n_classes)
        # self.n_classes = n_classes

    # def relocate(self):
    #     """
    #     Relocates the model to GPU if available, else to CPU.
    #     """
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     self.net_general = self.net_general.to(device)
    #     self.attention_net = self.attention_net.to(device)
    #     self.classifiers = self.classifiers.to(device)

    def forward(self, h, return_features=False, attention_only=False):
        """
        Forward pass of the model.

        Args:
            h (torch.Tensor): Input tensor
            attention_only (bool): Whether to return only attention weights
            return_features (bool): whether to return attention matrix M

        Returns:
            tuple: Tuple containing logits, predicted probabilities, predicted labels, attention weights, and attention weights before softmax

        """
        device = h.device

        h = self.net_general(h)
        N, C = h.shape

        # Multihead Input
        h = h.reshape(N, self.n_heads, C // self.n_heads)

        A = torch.empty(N, self.n_heads, 1).float().to(device)
        for nn in range(self.n_heads):
            a, _ = self.attention_net[nn](h[:, nn, :])
            A[:, nn, :] = a

        A = torch.transpose(A, 2, 0)  # KxheadsxN
        if attention_only:
            return A
        A_raw = A
        A_norm = torch.norm(A_raw, dim=1)
        A = F.softmax(A, dim=-1)  # softmax over N

        # Multihead Output
        M = torch.empty(1, self.size[1]).float().to(device)
        for nn in range(self.n_heads):
            m = torch.mm(A[:, nn, :], h[:, nn, :])
            M[:, self.step * nn: self.step * nn + self.step] = m

        # Singlehead Classification
        logits = self.classifiers(M)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)
        results_dict = {}

        if return_features:

            results_dict.update({'features': M})

        return logits, Y_prob, Y_hat, A_norm, results_dict