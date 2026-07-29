import torch
from torch import nn

def parameter_count(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

class FlattenMLP(nn.Module):
    def __init__(self,seq_len,input_dim,hidden=64,depth=2):
        super().__init__(); self.seq_len=seq_len; self.input_dim=input_dim; self.hidden=hidden; self.depth=depth
        dims=[seq_len*input_dim]+[hidden]*depth+[1]; layers=[]
        for i in range(len(dims)-2): layers += [nn.Linear(dims[i],dims[i+1]),nn.GELU()]
        layers += [nn.Linear(dims[-2],1)]; self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(x.flatten(1))
    def approx_macs(self):
        d=[self.seq_len*self.input_dim]+[self.hidden]*self.depth+[1]
        return sum(d[i]*d[i+1] for i in range(len(d)-1))

class MicroCell(nn.Module):
    def __init__(self,input_dim,state_dim,steps=2):
        super().__init__(); self.inp=nn.Linear(input_dim,state_dim); self.rec=nn.Linear(state_dim,state_dim,bias=False); self.gate=nn.Linear(input_dim+state_dim,state_dim); self.norm=nn.LayerNorm(state_dim); self.steps=steps
    def forward(self,x,s):
        drive=self.inp(x)
        for _ in range(self.steps):
            cand=torch.tanh(drive+self.rec(s)); g=torch.sigmoid(self.gate(torch.cat([x,s],-1))); s=self.norm(g*cand+(1-g)*s)
        return s

class MicroNetwork(nn.Module):
    def __init__(self,input_dim,circuits=4,state_dim=16,steps=2):
        super().__init__(); self.cells=nn.ModuleList([MicroCell(input_dim,state_dim,steps) for _ in range(circuits)]); total=circuits*state_dim
        self.mix=nn.Sequential(nn.Linear(total,total),nn.GELU(),nn.Linear(total,1)); self.circuits=circuits; self.state_dim=state_dim; self.steps=steps; self.input_dim=input_dim
    def forward(self,x):
        states=[torch.zeros(x.size(0),self.state_dim,device=x.device) for _ in self.cells]
        for t in range(x.size(1)): states=[c(x[:,t],s) for c,s in zip(self.cells,states)]
        return self.mix(torch.cat(states,-1))
    def approx_macs(self,seq_len):
        d,s,c,k=self.input_dim,self.state_dim,self.circuits,self.steps
        return seq_len*c*(d*s+k*(s*s+(d+s)*s))+(c*s)*(c*s)+(c*s)

def matched_models(seq_len,input_dim,circuits=4,state_dim=16,steps=2):
    micro=MicroNetwork(input_dim,circuits,state_dim,steps); target=parameter_count(micro); best=None
    for h in range(4,513):
        mlp=FlattenMLP(seq_len,input_dim,h,2); gap=abs(parameter_count(mlp)-target)
        if best is None or gap<best[0]: best=(gap,mlp)
    return best[1],micro
