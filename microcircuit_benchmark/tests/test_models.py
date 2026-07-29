import torch
from microbench.models import matched_models,parameter_count

def test_shapes_and_budget():
    mlp,micro=matched_models(10,2); x=torch.randn(8,10,2); assert mlp(x).shape==(8,1); assert micro(x).shape==(8,1); a=parameter_count(mlp); b=parameter_count(micro); assert abs(a-b)/max(a,b)<.15
