import numpy as np
import pytest
import random
from backend.models.membership_functions import GaussianMF, default_partition, degree
from backend.models.fuzzy_rules import infer
from backend.models.neuro_fuzzy import SugenoANFIS
from backend.models.genetic_programming import Node, evaluate, random_tree, evolve, crossover, mutate
from backend.models.genetic_algorithm import encode, decode, optimize

def test_gaussian_memberships_and_bounds():
    mf=GaussianMF(50,10)
    assert mf(50)==pytest.approx(1)
    assert mf(0)<.01
    with pytest.raises(ValueError): GaussianMF(101,4)
    assert 0<=degree(20,'low')<=1

def test_category_condition_and_independent_scores():
    values=dict(contamination=15,moisture=10,recyclability=95,hazard=4)
    metal=infer(values,'Metal'); paper=infer(values,'Paper')
    assert metal['suitability']>0 and metal['priority']>0
    assert any('R5' in r['rule'] for r in metal['active_rules'])
    assert not any('R5' in r['rule'] for r in paper['active_rules'])
    assert metal['suitability'] != metal['priority']

def test_anfis_hybrid_fit_and_metrics():
    X=np.array([[10,10,90,5],[80,90,80,5],[40,40,50,30],[20,15,80,10]],float)
    y=np.array([85,15,45,75],float); model=SugenoANFIS()
    history=model.fit(X,y,validation=(X,y),epochs=3)
    assert len(history['train_loss'])==3 and np.isfinite(model.predict(X)).all()
    assert model.metrics(X,y)['rmse']<1e-3

def test_gp_tree_execution_evolution():
    x={'contamination':10,'moisture':10,'recyclability':90,'hazard':5}
    node=Node('and',Node('>',Node('var',value='recyclability'),Node('const',value=70)),Node('<',Node('var',value='hazard'),Node('const',value=20)))
    assert evaluate(node,x) is True and node.size()==7
    assert random_tree().size()>0
    child_a,child_b=crossover(node,random_tree(),random.Random(4))
    assert child_a.size()>0 and child_b.size()>0
    assert mutate(node,random.Random(5)).size()>0
    result=evolve(population=8,generations=3,seed=3)
    assert result['generations']==3 and len(result['history'])==3
    assert result['data_label'].startswith('synthetic')

def test_ga_encoding_constraints_and_operators():
    p=decode([90,2,10,100,40,8])
    centers=[p[k].center for k in ('low','medium','high')]
    assert centers==sorted(centers) and min(np.diff(centers))>=8
    assert all(4<=p[k].sigma<=45 for k in p)
    assert len(encode())==6
    result=optimize(population=6,generations=2,seed=2,inversion=True,deletion=True)
    assert len(result['history'])==2 and result['inversion'] and result['deletion']

def test_teaching_csvs_match_formulas_and_are_used(tmp_path):
    from backend.models.teaching_data import load_gp_rows, load_ga_data, gp_label, ga_priority, write_csv
    rows,_=load_gp_rows()
    assert len(rows)==54 and all(r['y']==gp_label(*r['x'].values()) for r in rows)
    X,y,ga_rows,_=load_ga_data()
    assert X.shape==(108,4) and all(abs(t-ga_priority(*x))<1e-6 for x,t in zip(X,y))
    # A replacement CSV of real labels is picked up and reported as non-synthetic.
    custom=tmp_path/'real.csv'
    write_csv(custom,[{'contamination':c,'moisture':10,'recyclability':r,'hazard':5,'high_priority':int(r>50 and c<50),'synthetic':0} for c in (10,80) for r in (20,90)])
    result=evolve(population=6,generations=2,data_path=custom)
    assert result['data_label'].startswith('labeled examples (4 rows from real.csv)')

def test_teaching_csv_validation(tmp_path):
    from backend.models.teaching_data import load_gp_rows
    bad=tmp_path/'bad.csv'; bad.write_text('contamination,moisture,recyclability,hazard,high_priority\n10,10,90,5,7\n')
    with pytest.raises(ValueError, match='0 or 1'): load_gp_rows(bad)
    with pytest.raises(FileNotFoundError): load_gp_rows(tmp_path/'missing.csv')
