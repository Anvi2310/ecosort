"""Compact first-order Sugeno ANFIS with hybrid premise/consequent fitting."""
import numpy as np
from .membership_functions import default_partition

class SugenoANFIS:
    def __init__(self, partition=None, seed=17):
        self.partition = partition or default_partition()
        rng = np.random.default_rng(seed)
        # 3 selected rules avoid a 3^4 Cartesian rule explosion.
        self.antecedents = [(0,"low"),(1,"low"),(2,"high"),(3,"low")], [(0,"high"),(2,"high"),(3,"high")], [(0,"low"),(1,"low"),(3,"low")]
        self.consequents = rng.normal(0, .02, (3,5))
        self.history = {"train_loss": [], "validation_loss": []}

    def _fire(self, X):
        X=np.asarray(X,dtype=float); out=[]
        for rule in self.antecedents:
            vals=[self.partition[label](X[:,idx]) for idx,label in rule]
            out.append(np.prod(vals,axis=0))
        F=np.asarray(out).T
        return F/(F.sum(axis=1,keepdims=True)+1e-12)

    def predict(self, X):
        X=np.asarray(X,dtype=float); W=self._fire(X)
        design=np.concatenate([W[:,i:i+1]*np.c_[X,np.ones(len(X))] for i in range(W.shape[1])],axis=1)
        return design @ self.consequents.reshape(-1)

    def fit(self, X, y, validation=None, epochs=15, learning_rate=.01):
        X=np.asarray(X,dtype=float); y=np.asarray(y,dtype=float)
        if X.ndim!=2 or X.shape[1]!=4 or len(X)!=len(y) or len(X)==0: raise ValueError("Expected non-empty X[n,4] and y[n]")
        X=np.clip(X,0,100)
        Xv,yv=(X,y) if validation is None else (np.asarray(validation[0],float),np.asarray(validation[1],float))
        for _ in range(max(1,int(epochs))):
            W=self._fire(X); design=np.concatenate([W[:,i:i+1]*np.c_[X,np.ones(len(X))] for i in range(W.shape[1])],axis=1)
            beta=np.linalg.lstsq(design,y,rcond=None)[0]; self.consequents=beta.reshape(3,5)
            err=self.predict(X)-y; self.history["train_loss"].append(float(np.mean(err**2)))
            self.history["validation_loss"].append(float(np.mean((self.predict(Xv)-yv)**2)))
            # Finite-difference premise update (bounded, deterministic) on Gaussian centers/widths.
            for mf in self.partition.values():
                for attr,lo,hi in (("center",0,100),("sigma",1,60)):
                    original=getattr(mf,attr); eps=.2
                    setattr(mf,attr,min(hi,original+eps)); plus=np.mean((self.predict(X)-y)**2)
                    setattr(mf,attr,max(lo,original-eps)); minus=np.mean((self.predict(X)-y)**2)
                    grad=(plus-minus)/(2*eps); setattr(mf,attr,float(np.clip(original-learning_rate*grad,lo,hi)))
        return self.history

    def metrics(self,X,y):
        y=np.asarray(y,float); pred=self.predict(X); mae=float(np.mean(np.abs(pred-y))); rmse=float(np.sqrt(np.mean((pred-y)**2)))
        denom=float(np.sum((y-y.mean())**2)); r2=1-float(np.sum((pred-y)**2))/(denom or 1)
        return {"mae":mae,"rmse":rmse,"r2":r2}
