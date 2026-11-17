"""Small executable tree GP for evolving binary priority rules."""
from dataclasses import dataclass
import copy
import random

FEATURES=("contamination","moisture","recyclability","hazard")
@dataclass
class Node:
    op:str
    left:object=None
    right:object=None
    value:object=None
    def run(self,x):
        if self.op=="var": return float(x[self.value])
        if self.op=="const": return float(self.value)
        a=self.left.run(x)
        if self.op=="not": return not bool(a)
        b=self.right.run(x)
        return {"and":lambda:bool(a) and bool(b),"or":lambda:bool(a) or bool(b),
                ">":lambda:a>b,"<":lambda:a<b,">=":lambda:a>=b,"<=":lambda:a<=b}[self.op]()
    def size(self): return 1+(self.left.size() if self.left else 0)+(self.right.size() if self.right else 0)
    def text(self):
        if self.op=="var": return self.value
        if self.op=="const": return str(self.value)
        if self.op=="not": return f"NOT ({self.left.text()})"
        return f"({self.left.text()} {self.op.upper()} {self.right.text()})"

def random_tree(rng=random,depth=3):
    if depth<=0 or rng.random()<.3: return Node("var",value=rng.choice(FEATURES)) if rng.random()<.8 else Node("const",value=rng.choice([20,40,60,80]))
    op=rng.choice(["and","or",">","<",">=","<="])
    return Node(op,random_tree(rng,depth-1),random_tree(rng,depth-1))

def _paths(node,path=()):
    out=[path]
    if node.left is not None: out+=_paths(node.left,path+('left',))
    if node.right is not None: out+=_paths(node.right,path+('right',))
    return out

def _at(node,path):
    for part in path: node=getattr(node,part)
    return node

def _replace(node,path,replacement):
    if not path: return copy.deepcopy(replacement)
    parent=_at(node,path[:-1]); setattr(parent,path[-1],copy.deepcopy(replacement)); return node

def crossover(first,second,rng=random):
    """Exchange randomly selected subtrees, returning independent offspring."""
    a,b=copy.deepcopy(first),copy.deepcopy(second)
    pa=rng.choice(_paths(a)); pb=rng.choice(_paths(b))
    sa,sb=copy.deepcopy(_at(a,pa)),copy.deepcopy(_at(b,pb))
    return _replace(a,pa,sb),_replace(b,pb,sa)

def mutate(tree,rng=random,max_depth=2):
    """Replace one randomly selected subtree with a freshly generated subtree."""
    result=copy.deepcopy(tree); path=rng.choice(_paths(result))
    return _replace(result,path,random_tree(rng,max_depth))

def evaluate(root,x):
    try: return bool(root.run(x))
    except (TypeError,ValueError,KeyError): return False

def evolve(rows=None,population=32,generations=12,seed=19,data_path=None):
    rng=random.Random(seed)
    # Labeled examples come from a CSV (data/synthetic/gp_priority_labels.csv by default).
    from .teaching_data import load_gp_rows, describe
    if rows is None:
        rows,path=load_gp_rows(data_path); data_label=describe(rows,path)
    else:
        data_label="provided labeled examples"
    def fit(t):
        acc=sum(evaluate(t,z["x"])==bool(z["y"]) for z in rows)/len(rows)
        return acc-.002*min(t.size(),100)
    pop=[random_tree(rng) for _ in range(population)]; history=[]; best=None
    for _ in range(generations):
        ranked=sorted(pop,key=fit,reverse=True); best=ranked[0]; history.append(fit(best)); nxt=ranked[:2]
        while len(nxt)<population:
            parents=rng.sample(ranked[:max(2,population//2)],2)
            child,_=crossover(parents[0],parents[1],rng)
            if rng.random()<.25: child=mutate(child,rng,2)
            nxt.append(child)
        pop=nxt
    return {"best_rule":best.text(),"fitness":fit(best),"accuracy":sum(evaluate(best,z["x"])==bool(z["y"]) for z in rows)/len(rows),"complexity":best.size(),"history":history,"generations":generations,"data_label":data_label,"_tree":best}
