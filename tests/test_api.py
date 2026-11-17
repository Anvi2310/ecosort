import io
import pytest
from app import create_app

@pytest.fixture
def client(tmp_path,monkeypatch):
    import config
    import backend.database.store as store
    monkeypatch.setattr(config,'DATABASE_PATH',tmp_path/'test.sqlite3')
    monkeypatch.setattr(store,'DATABASE_PATH',tmp_path/'test.sqlite3')
    app=create_app(); app.config['TESTING']=True
    return app.test_client()

def test_health_membership_and_invalid_analysis(client):
    assert client.get('/api/health').status_code==200
    assert set(client.get('/api/membership').get_json())=={'low','medium','high'}
    res=client.post('/api/analyze',json={'category':'Plastic','contamination':200,'moisture':0,'recyclability':50,'hazard':0})
    assert res.status_code==400

def test_analysis_history_export(client):
    response=client.post('/api/analyze',json={'category':'Plastic','contamination':20,'moisture':10,'recyclability':90,'hazard':5})
    assert response.status_code==200
    assert response.get_json()['active_rules']
    assert len(client.get('/api/history').get_json())==1
    assert b'text/csv' in client.get('/api/export.csv').content_type.encode()
    report=client.get('/api/report.pdf')
    assert report.status_code==200 and report.data.startswith(b'%PDF-1.4')

def test_classifier_rejects_bad_file(client):
    res=client.post('/api/classify',data={'image':(io.BytesIO(b'not image'),'bad.gif')},content_type='multipart/form-data')
    assert res.status_code==415

def test_anfis_training_requires_and_uses_labeled_split(client):
    samples=[{'contamination':i*8,'moisture':i*5,'recyclability':100-i*7,'hazard':i*4,'priority':90-i*6} for i in range(10)]
    result=client.post('/api/anfis/train',json={'samples':samples,'epochs':2})
    assert result.status_code==200
    body=result.get_json()
    assert body['training_samples']+body['validation_samples']+body['test_samples']==10
    assert set(body['metrics'])=={'mae','rmse','r2'}
    assert client.get('/api/anfis/status').get_json()['status']=='trained'
