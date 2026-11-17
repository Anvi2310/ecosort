"""SQLite persistence for analysis and optimizer run history."""
import sqlite3
from contextlib import contextmanager
from config import DATABASE_PATH

SCHEMA='''
CREATE TABLE IF NOT EXISTS WasteAnalysis(id INTEGER PRIMARY KEY, image_ref TEXT, predicted_category TEXT, corrected_category TEXT, confidence REAL, contamination REAL, moisture REAL, recyclability REAL, hazard REAL, suitability REAL, priority REAL, recommendation TEXT, model_version TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS ClassificationResults(id INTEGER PRIMARY KEY, analysis_id INTEGER, category TEXT, confidence REAL, status TEXT, FOREIGN KEY(analysis_id) REFERENCES WasteAnalysis(id));
CREATE TABLE IF NOT EXISTS NeuroFuzzyPredictions(id INTEGER PRIMARY KEY, analysis_id INTEGER, suitability REAL, priority REAL, explanation TEXT, FOREIGN KEY(analysis_id) REFERENCES WasteAnalysis(id));
CREATE TABLE IF NOT EXISTS GeneticProgrammingRuns(id INTEGER PRIMARY KEY, best_rule TEXT, fitness REAL, payload TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS GeneticAlgorithmRuns(id INTEGER PRIMARY KEY, fitness REAL, payload TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS ConditionLabels(id INTEGER PRIMARY KEY, analysis_id INTEGER, image_ref TEXT, category TEXT, item TEXT, answers TEXT, flags TEXT, visual TEXT, contamination REAL, moisture REAL, hazard REAL, user_adjusted INTEGER, created_at TEXT DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY(analysis_id) REFERENCES WasteAnalysis(id));
CREATE TABLE IF NOT EXISTS OptimizationHistory(id INTEGER PRIMARY KEY, kind TEXT, generation INTEGER, fitness REAL, run_id INTEGER);
'''

@contextmanager
def connect():
    DATABASE_PATH.parent.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(DATABASE_PATH); db.row_factory=sqlite3.Row
    try:
        db.executescript(SCHEMA)
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

def add_analysis(data):
    with connect() as db:
        cur=db.execute('INSERT INTO WasteAnalysis(image_ref,predicted_category,corrected_category,confidence,contamination,moisture,recyclability,hazard,suitability,priority,recommendation,model_version) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)', tuple(data.get(k) for k in ('image_ref','predicted_category','corrected_category','confidence','contamination','moisture','recyclability','hazard','suitability','priority','recommendation','model_version')))
        return cur.lastrowid

def recent(limit=100):
    with connect() as db: return [dict(r) for r in db.execute('SELECT * FROM WasteAnalysis ORDER BY id DESC LIMIT ?', (limit,))]

def insert_run(table, payload, best_rule=None):
    import json
    with connect() as db:
        if table=='GeneticProgrammingRuns': cur=db.execute('INSERT INTO GeneticProgrammingRuns(best_rule,fitness,payload) VALUES(?,?,?)',(best_rule,payload['fitness'],json.dumps(payload)))
        else: cur=db.execute('INSERT INTO GeneticAlgorithmRuns(fitness,payload) VALUES(?,?)',(payload['best_fitness'],json.dumps(payload)))
        rid=cur.lastrowid
        for gen,fit in enumerate(payload.get('history',[])): db.execute('INSERT INTO OptimizationHistory(kind,generation,fitness,run_id) VALUES(?,?,?,?)',(table,gen,float(fit),rid))
        return rid

def add_condition_label(analysis_id, data):
    """Store the evidence behind an analysis so confirmed photos become a real labelled dataset."""
    import json
    with connect() as db:
        db.execute('INSERT INTO ConditionLabels(analysis_id,image_ref,category,item,answers,flags,visual,contamination,moisture,hazard,user_adjusted) VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                   (analysis_id, data.get('image_ref'), data.get('category'), data.get('item'), json.dumps(data.get('answers') or {}),
                    json.dumps(data.get('flags') or []), json.dumps(data.get('visual')), data['contamination'], data['moisture'], data['hazard'], int(bool(data.get('user_adjusted')))))

def condition_labels(limit=100000):
    with connect() as db: return [dict(r) for r in db.execute('SELECT * FROM ConditionLabels ORDER BY id LIMIT ?', (limit,))]
