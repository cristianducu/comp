from microbench.tasks import all_tasks

def test_tasks():
    for spec,splits in all_tasks(0):
        assert len(splits)==3
        for ds in splits:
            x,y=ds[0]; assert x.ndim==2 and x.shape[-1]==spec.input_dim and y.shape==(1,)
