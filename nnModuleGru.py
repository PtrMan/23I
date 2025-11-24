import torch
import torch.nn as nn
import math

# minimal gated unit
#
# https://en.wikipedia.org/wiki/Gated_recurrent_unit
class GatedRecurrentUnit_MinimalGatedUnit(torch.nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # --- Parameters for Forget Gate (f) ---
        # Wf: Weights for input x
        self.Wf = nn.Parameter(torch.Tensor(input_size, hidden_size)).cuda()
        # Uf: Weights for previous hidden state
        self.Uf = nn.Parameter(torch.Tensor(hidden_size, hidden_size)).cuda()
        # bf: Bias
        self.bf = nn.Parameter(torch.Tensor(hidden_size)).cuda()
        
        # --- Parameters for Candidate Activation (h_hat) ---
        # Wh: Weights for input x
        self.Wh = nn.Parameter(torch.Tensor(input_size, hidden_size)).cuda()
        # Uh: Weights for the masked previous hidden state
        self.Uh = nn.Parameter(torch.Tensor(hidden_size, hidden_size)).cuda()
        # bh: Bias
        self.bh = nn.Parameter(torch.Tensor(hidden_size)).cuda()
        
        # Internal state storage
        self.ht = None
        
        # Initialize parameters
        self.resetParameters()

    def resetParameters(self):
        torch.nn.init.kaiming_uniform_(self.Wf)


        v0 = torch.ones(self.hidden_size)
        v0 = torch.diag(v0)

        v1 = torch.Tensor(self.hidden_size, self.hidden_size)
        torch.nn.init.normal_(v1, mean=0.0, std=0.01)

        self.Uf = v0 + v1
        self.Uf = self.Uf.cuda()

        #torch.nn.init.kaiming_uniform_(self.Uf)
        
        # the gate to be init with ~ -1.5 so that the gate is closed at the beginning
        torch.nn.init.normal_(self.bf, mean=-1.5, std=0.05)

        torch.nn.init.kaiming_uniform_(self.Wh)



        #torch.nn.init.kaiming_uniform_(self.Uh)

        v0 = torch.ones(self.hidden_size)
        v0 = torch.diag(v0)

        v1 = torch.Tensor(self.hidden_size, self.hidden_size)
        torch.nn.init.normal_(v1, mean=0.0, std=0.01)

        self.Uh = v0 + v1
        self.Uh = self.Uh.cuda()






        torch.nn.init.normal_(self.bh, mean=0.0, std=0.02)

        

    def resetHiddenstate(self):
        self.ht = None

    def step(self, x):
        """
        Forward step for a single time step.
        x shape: (batch_size, input_size)
        """
        batch_size = x.size(0)
        
        # Initialize hidden state if it doesn't exist or batch size changed
        if self.ht is None or self.ht.size(0) != batch_size:
            self.ht = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        htPrevious = self.ht

        ft = torch.nn.functional.sigmoid( x @ self.Wf + htPrevious @ self.Uf + self.bf )

        helper = ft * htPrevious
        htRoof = torch.nn.functional.tanh( x @ self.Wh + helper @ self.Uh + self.bh )
        
        self.ht = (1 - ft) * htPrevious + ft * htRoof

        return self.ht
