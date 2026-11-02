"""Host metadata only; no tensor values, GPU operations or model changes."""
import os

def readback(model):
    rows=[]
    for name,module in model.named_modules():
        weight=getattr(module,'weight',None)
        if getattr(weight,'ndim',None)==2:
            method=getattr(module,'quant_method',None)
            rows.append(dict(name=name,type=type(module).__name__,shape=list(weight.shape),
                             dtype=str(weight.dtype),stride=list(weight.stride()),
                             quant_method=type(method).__name__,
                             block_quant=bool(getattr(method,'block_quant',False))))
    return dict(case=os.environ.get('CODEX_AR87_CASE'),boundary=os.environ.get('CODEX_AR86_BOUNDARY'),
                census=os.environ.get('CODEX_AR86_CENSUS'),projections=rows,
                graph_directory=os.environ.get('CODEX_AR87_GRAPH_DIR'),
                evidence='captured forward graph and static module shapes; no performance measurement')
