"""Real-valued GA for constrained fuzzy center/width optimization."""
import numpy as np
from .membership_functions import default_partition, GaussianMF

def encode(partition=None):
    p=partition or default_partition()
    return np.array([[p[k].center,p[k].sigma] for k in ("low","medium","high")],float).ravel()

def decode(chrom):
    a=np.asarray(chrom,float).reshape(3,2)
    centers=np.clip(a[:,0],0,100); centers.sort()
    for i in range(1,3): centers[i]=max(centers[i],centers[i-1]+8)
    centers=np.clip(centers,0,100)
    return {k:GaussianMF(c,float(np.clip(s,4,45))) for k,(c,s) in zip(("low","medium","high"),zip(centers,a[:,1]))}

def optimize(population=36,generations=20,seed=23, inversion=False,deletion=False,data_path=None):
    rng=np.random.default_rng(seed)
    # Priority targets come from a CSV (data/synthetic/ga_priority_targets.csv by default).
    from .teaching_data import load_ga_data, describe
    X,y,rows,path=load_ga_data(data_path)
    base=encode(); pop=np.tile(base,(population,1))+rng.normal(0,8,(population,6)); history=[]
    def score(ch):
        part=decode(ch); centers=np.array([part[k].center for k in ("low","medium","high")]); se=np.mean((np.sort(centers)-[15,50,85])**2)/10000
        # Gaussian quality is fit to transparent synthetic priority targets plus ordered-set regularization.
        from .neuro_fuzzy import SugenoANFIS
        model=SugenoANFIS(part); model.fit(X,y,epochs=1,learning_rate=0); pred=model.predict(X)
        return 1/(1+np.mean((pred-y)**2)/1000+se)
    for _ in range(generations):
        fit=np.array([score(ch) for ch in pop]); ix=np.argsort(fit)[::-1]; history.append(float(fit[ix[0]])); nxt=[pop[ix[0]].copy()]
        while len(nxt)<population:
            p1,p2=pop[rng.choice(ix[:max(2,population//3)],2,replace=False)]
            mask=rng.random(6)<.5; child=np.where(mask,p1,p2)+rng.normal(0,3,6)
            if inversion and rng.random()<.1: child=child.reshape(3,2)[::-1].flatten()
            if deletion and rng.random()<.1: child[1::2]=np.maximum(4,child[1::2]*.9)
            nxt.append(child)
        pop=np.array(nxt)
    scores=np.array([score(ch) for ch in pop]); best=pop[int(scores.argmax())]
    return {"initial":base.tolist(),"best":best.tolist(),"initial_fitness":score(base),"best_fitness":float(scores.max()),"history":history,"parameters":decode(best) and {k:{"center":v.center,"sigma":v.sigma} for k,v in decode(best).items()},"inversion":inversion,"deletion":deletion,"fitness_data":describe(rows,path)}
