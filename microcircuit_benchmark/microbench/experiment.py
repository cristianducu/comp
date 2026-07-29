from pathlib import Path
import random,time,numpy as np,pandas as pd,torch
from torch import nn
from torch.utils.data import DataLoader
from .models import matched_models,parameter_count
from .tasks import all_tasks

def seed_all(s): random.seed(s); np.random.seed(s); torch.manual_seed(s)
@torch.no_grad()
def evaluate(model,loader,device):
    model.eval(); loss_fn=nn.BCEWithLogitsLoss(reduction='sum'); loss=correct=total=0
    for x,y in loader:
        x,y=x.to(device),y.to(device); z=model(x); loss+=loss_fn(z,y).item(); p=torch.sigmoid(z)>=.5; correct+=(p==y.bool()).sum().item(); total+=y.numel()
    return loss/total,correct/total

def train(model,splits,epochs,batch,lr,seed,device):
    seed_all(seed); tr,va,te=splits; tl=DataLoader(tr,batch_size=batch,shuffle=True); vl=DataLoader(va,batch_size=512); xl=DataLoader(te,batch_size=512)
    model.to(device); opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4); loss_fn=nn.BCEWithLogitsLoss(); best=-1; state=None; best_epoch=0; curve=[]; start=time.perf_counter()
    for epoch in range(1,epochs+1):
        model.train()
        for x,y in tl:
            x,y=x.to(device),y.to(device); opt.zero_grad(set_to_none=True); loss=loss_fn(model(x),y); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1); opt.step()
        _,a=evaluate(model,vl,device); curve.append((epoch,a))
        if a>best: best=a; best_epoch=epoch; state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    secs=time.perf_counter()-start; model.load_state_dict(state); l,a=evaluate(model,xl,device)
    return best,a,l,secs,best_epoch,curve

def run_benchmark(out=Path('results'),epochs=300,seeds=5,batch=128,device=None):
    out=Path(out); out.mkdir(parents=True,exist_ok=True); device=device or ('cuda' if torch.cuda.is_available() else 'cpu'); rows=[]; curves=[]
    for seed in range(seeds):
        for spec,splits in all_tasks(seed):
            mlp,micro=matched_models(spec.sequence_length,spec.input_dim)
            for name,model,lr in [('mlp',mlp,2e-3),('microcircuit',micro,1e-3)]:
                bv,ta,tl,secs,be,curve=train(model,splits,epochs,batch,lr,seed,device)
                macs=model.approx_macs() if name=='mlp' else model.approx_macs(spec.sequence_length)
                rows.append(dict(task=spec.name,model=name,seed=seed,parameters=parameter_count(model),approx_macs_per_example=macs,best_val_accuracy=bv,test_accuracy=ta,test_loss=tl,seconds=secs,epoch_of_best=be))
                curves += [dict(task=spec.name,model=name,seed=seed,epoch=e,val_accuracy=a) for e,a in curve]
    runs=pd.DataFrame(rows); cv=pd.DataFrame(curves); runs.to_csv(out/'runs.csv',index=False); cv.to_csv(out/'curves.csv',index=False)
    summary=runs.groupby(['task','model']).agg(test_accuracy_mean=('test_accuracy','mean'),test_accuracy_std=('test_accuracy','std'),parameters_mean=('parameters','mean'),macs_mean=('approx_macs_per_example','mean'),seconds_mean=('seconds','mean'),epoch_of_best_mean=('epoch_of_best','mean')).reset_index(); summary.to_csv(out/'summary.csv',index=False)
    write_report(summary,out); plot(cv,out); return runs,summary

def write_report(summary,out):
    pivot=summary.pivot(index="task",columns="model",values="test_accuracy_mean")
    wins=[]
    for t in pivot.index:
        if {"mlp","microcircuit"}.issubset(pivot.columns):
            d=float(pivot.loc[t,"microcircuit"]-pivot.loc[t,"mlp"])
            if d>=.03:
                wins.append((t,d))
    memory=any(("delayed" in t or "copy" in t) for t,_ in wins)
    supported=len(wins)>=2 and memory
    verdict="PROVISIONALLY SUPPORTED" if supported else "NOT SUPPORTED BY THIS RUN"
    text = (
        "# Microcircuit Benchmark Report\n\n"
        "## Decision rule\n"
        "Microcircuit must beat the matched MLP by at least 3 percentage points on at least two tasks, including a delayed-memory task.\n\n"
        "## Results\n\n"
        + summary.to_markdown(index=False)
        + f"\n\n## Verdict\n\n**{verdict}.**\n\nWins: "
        + (", ".join(f"{t} (+{d:.3f})" for t,d in wins) or "none")
        + "\n"
    )
    (out/"report.md").write_text(text)

def plot(curves,out):
    import matplotlib.pyplot as plt
    tasks=list(curves.task.unique()); fig,axs=plt.subplots(len(tasks),1,figsize=(9,4*len(tasks)),squeeze=False)
    for ax,t in zip(axs[:,0],tasks):
        s=curves[curves.task==t]
        for m in s.model.unique():
            g=s[s.model==m].groupby('epoch').val_accuracy; mean=g.mean(); std=g.std().fillna(0); ax.plot(mean.index,mean.values,label=m); ax.fill_between(mean.index,mean-std,mean+std,alpha=.2)
        ax.set_title(t); ax.set_ylim(.45,1.01); ax.set_xlabel('Epoch'); ax.set_ylabel('Validation accuracy'); ax.legend()
    fig.tight_layout(); fig.savefig(out/'learning_curves.png',dpi=160); plt.close(fig)
