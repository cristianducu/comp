from dataclasses import dataclass
import torch
from torch.utils.data import TensorDataset

@dataclass(frozen=True)
class TaskSpec:
    name: str
    sequence_length: int
    input_dim: int = 2

def split(x,y,seed):
    g=torch.Generator().manual_seed(seed)
    idx=torch.randperm(len(x),generator=g)
    a=int(.70*len(x)); b=int(.85*len(x))
    return TensorDataset(x[idx[:a]],y[idx[:a]]),TensorDataset(x[idx[a:b]],y[idx[a:b]]),TensorDataset(x[idx[b:]],y[idx[b:]])

def parity(n=4096,bits=12,seed=0):
    g=torch.Generator().manual_seed(seed); r=torch.randint(0,2,(n,bits),generator=g).float()
    x=torch.stack([r,torch.ones_like(r)],-1); y=r.sum(1).remainder(2).unsqueeze(1)
    return TaskSpec(f'parity_{bits}',bits),split(x,y,seed+100)

def delayed_xor(n=4096,delay=20,seed=0):
    g=torch.Generator().manual_seed(seed); a=torch.randint(0,2,(n,1),generator=g).float(); b=torch.randint(0,2,(n,1),generator=g).float()
    x=torch.zeros(n,delay+2,2); x[:,0,0]=a[:,0]; x[:,1,0]=b[:,0]; x[:,:2,1]=1; x[:,2:,0]=torch.randint(0,2,(n,delay),generator=g).float()
    return TaskSpec(f'delayed_xor_{delay}',delay+2),split(x,(a!=b).float(),seed+200)

def copy_first(n=4096,delay=30,seed=0):
    g=torch.Generator().manual_seed(seed); y=torch.randint(0,2,(n,1),generator=g).float(); x=torch.zeros(n,delay+1,2)
    x[:,0,0]=y[:,0]; x[:,0,1]=1; x[:,1:,0]=torch.randint(0,2,(n,delay),generator=g).float()
    return TaskSpec(f'copy_first_{delay}',delay+1),split(x,y,seed+300)

def composition(n=4096,length=16,seed=0):
    g=torch.Generator().manual_seed(seed); r=torch.randint(0,2,(n,length),generator=g).float(); x=torch.stack([r,torch.ones_like(r)],-1)
    y=((r[:,0]!=r[:,1]) & (r[:,-2]!=r[:,-1])).float().unsqueeze(1)
    return TaskSpec(f'composition_{length}',length),split(x,y,seed+400)

def all_tasks(seed=0):
    return [parity(seed=seed),delayed_xor(seed=seed),copy_first(seed=seed),composition(seed=seed)]
