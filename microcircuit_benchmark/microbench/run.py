import argparse
from pathlib import Path
from .experiment import run_benchmark

def main():
    p=argparse.ArgumentParser(); p.add_argument('--epochs',type=int,default=300); p.add_argument('--seeds',type=int,default=5); p.add_argument('--batch-size',type=int,default=128); p.add_argument('--output',type=Path,default=Path('results')); p.add_argument('--device',choices=['cpu','cuda'],default=None); a=p.parse_args()
    _,s=run_benchmark(a.output,a.epochs,a.seeds,a.batch_size,a.device); print(s.to_string(index=False)); print(f'Full report: {a.output / "report.md"}')
if __name__=='__main__': main()
