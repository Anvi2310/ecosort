"""EcoSort Flask application and JSON API."""
from flask import Flask, jsonify, request, send_from_directory, Response
from werkzeug.utils import secure_filename
from pathlib import Path
import csv, io, json, uuid
from config import SECRET_KEY, MAX_UPLOAD_BYTES, ALLOWED_IMAGE_EXTENSIONS, CATEGORIES, MODEL_VERSION, ROOT
from backend.database.store import add_analysis, recent, insert_run, connect, add_condition_label, condition_labels
from backend.services.condition_estimator import catalog as condition_catalog, estimate as estimate_condition
from backend.models.condition_model import ConditionModel
from backend.models.fuzzy_rules import infer
from backend.models.membership_functions import partition_payload
from backend.models.genetic_programming import evolve
from backend.models.genetic_programming import evaluate
from backend.models.genetic_algorithm import optimize, decode
from backend.models.membership_functions import default_partition
from backend.models.neuro_fuzzy import SugenoANFIS
from backend.models.waste_classifier import WasteClassifier

def bounded(payload,key,default,lo,hi):
    """Read an integer request parameter and keep it in a safe range (protects a public server)."""
    try: value=int(payload.get(key,default))
    except (TypeError,ValueError): raise ValueError(f"{key} must be a whole number") from None
    if not lo<=value<=hi: raise ValueError(f"{key} must be between {lo} and {hi}")
    return value

def create_app():
    app=Flask(__name__,static_folder="frontend",static_url_path="")
    app.config.update(SECRET_KEY=SECRET_KEY,MAX_CONTENT_LENGTH=MAX_UPLOAD_BYTES)
    classifier=WasteClassifier(); condition_model=ConditionModel(); uploads=ROOT/"instance"/"uploads"; uploads.mkdir(parents=True,exist_ok=True)
    state={"partition":default_partition(),"gp_tree":None,"anfis":None,"anfis_metrics":None}
    @app.errorhandler(413)
    def too_large(_): return jsonify(error="Image exceeds upload limit"),413
    @app.errorhandler(400)
    def bad_request(e): return jsonify(error=str(e.description)),400
    @app.get("/api/health")
    def health(): return jsonify(status="ok",project="EcoSort",classifier=classifier.status,condition_model=condition_model.status)
    @app.get("/api/membership")
    def membership(): return jsonify(partition_payload(state['partition']))
    @app.get("/api/history")
    def history(): return jsonify(recent())
    @app.get("/api/analytics")
    def analytics():
        rows=recent(10000); by={k:0 for k in CATEGORIES}
        for row in rows:
            key=row.get("corrected_category") or row.get("predicted_category")
            if key in by: by[key]+=1
        n=len(rows)
        return jsonify(total=n,category_distribution=by,recyclable_percent=round(100*sum(r['suitability']>=50 for r in rows)/(n or 1),1),high_priority=sum(r['priority']>=70 for r in rows))
    @app.get("/api/training/status")
    def training_status(): return jsonify(status="not_trained",classifier=classifier.status,checkpoint="backend/models/weights/waste_mobilenet_v2.pth",dataset="data/processed/images",message="No EcoSort fine-tuned checkpoint is assumed; add the dataset and run the training script.")
    def validated_inputs(payload):
        category=payload.get("category")
        if category not in CATEGORIES: raise ValueError("category must be one of: "+", ".join(CATEGORIES))
        result={}
        for k in ("contamination","moisture","recyclability","hazard"):
            v=payload.get(k)
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not 0<=v<=100: raise ValueError(f"{k} must be a number from 0 to 100")
            result[k]=float(v)
        return category,result
    def uploaded(ref):
        """Resolve an image_ref to a file inside uploads/, rejecting path tricks."""
        if not ref or secure_filename(ref)!=ref: return None
        path=uploads/ref
        return path if path.is_file() else None
    @app.get("/api/condition/catalog")
    def condition_catalog_route(): return jsonify({**condition_catalog(),"condition_model":condition_model.status})
    @app.post("/api/condition/estimate")
    def condition_estimate():
        payload=request.get_json(silent=True) or {}
        image=uploaded(payload.get("image_ref"))
        visual=condition_model.predict(image) if image else None
        try: result=estimate_condition(payload.get("category"),payload.get("item"),payload.get("answers"),payload.get("flags"),visual)
        except (ValueError,TypeError,StopIteration) as exc: return jsonify(error=str(exc)),400
        return jsonify(**result,visual=visual,condition_model=condition_model.status)
    @app.get("/api/condition/labels.csv")
    def condition_labels_csv():
        """Human-confirmed condition labels; with the images in instance/uploads this is a trainable manifest."""
        out=io.StringIO(); w=csv.writer(out); w.writerow(['file','group','material','item','contamination','moisture','hazard','user_adjusted','answers','synthetic'])
        for r in condition_labels():
            if r['image_ref']: w.writerow([r['image_ref'],r['image_ref'],r['category'],r['item'],r['contamination'],r['moisture'],r['hazard'],r['user_adjusted'],r['answers'],0])
        return Response(out.getvalue(),mimetype='text/csv',headers={'Content-Disposition':'attachment; filename=manifest.csv'})
    @app.post("/api/classify")
    def classify():
        if "image" not in request.files: return jsonify(error="multipart image is required"),400
        f=request.files['image']; ext=Path(secure_filename(f.filename or "")).suffix.lower()
        if ext not in ALLOWED_IMAGE_EXTENSIONS: return jsonify(error="supported formats: jpg, jpeg, png, webp"),415
        path=uploads/(uuid.uuid4().hex+ext); f.save(path)
        try:
            from PIL import Image
            with Image.open(path) as im: im.verify()
            result=classifier.predict(path); result['image_ref']=path.name
            result['visual_condition']=condition_model.predict(path); result['condition_model']=condition_model.status
            return jsonify(result)
        except Exception as exc:
            path.unlink(missing_ok=True); return jsonify(error=f"Invalid image: {exc}"),400
    @app.post("/api/analyze")
    def analyze():
        payload=request.get_json(silent=True) or {}
        try: category,values=validated_inputs(payload)
        except ValueError as exc: return jsonify(error=str(exc)),400
        result=infer(values,category,state['partition'])
        if state['gp_tree'] is not None:
            gp_match=evaluate(state['gp_tree'],values)
            result['priority']=round(max(0,min(100,result['priority']+(4 if gp_match else -4))),2)
            result['active_rules'].append({"rule":"GP evolved condition: "+state['gp_tree'].text(),"strength":1.0,"suitability":result['suitability'],"priority":result['priority']})
            result['gp_rule_applied']=gp_match
        if state['anfis'] is not None:
            learned=float(state['anfis'].predict([[values[k] for k in ('contamination','moisture','recyclability','hazard')]])[0])
            result['priority']=round(max(0,min(100,.5*result['priority']+.5*learned)),2)
            result['anfis_priority']=round(learned,2)
        data={**values,"predicted_category":payload.get("predicted_category",category),"corrected_category":payload.get("corrected_category"),"confidence":payload.get("confidence"),"image_ref":payload.get("image_ref"),"suitability":result['suitability'],"priority":result['priority'],"recommendation":result['recommendation'],"model_version":MODEL_VERSION}
        analysis_id=add_analysis(data)
        evidence=payload.get("condition")
        if isinstance(evidence,dict):
            add_condition_label(analysis_id,{**evidence,"image_ref":payload.get("image_ref"),"category":category,**{k:values[k] for k in ("contamination","moisture","hazard")}})
        with connect() as db:
            db.execute('INSERT INTO NeuroFuzzyPredictions(analysis_id,suitability,priority,explanation) VALUES(?,?,?,?)',(analysis_id,result['suitability'],result['priority'],json.dumps(result['active_rules'])))
        return jsonify(id=analysis_id,category=category,inputs=values,**result,mode="expert fuzzy inference; not dataset-trained")
    @app.post("/api/gp/evolve")
    def gp_run():
        payload=request.get_json(silent=True) or {}
        try: result=evolve(population=bounded(payload,'population',32,4,100),generations=bounded(payload,'generations',12,1,100))
        except (ValueError,FileNotFoundError) as exc: return jsonify(error=f"GP: {exc}"),400
        state['gp_tree']=result.pop('_tree')
        insert_run('GeneticProgrammingRuns',result,result['best_rule']); return jsonify(result)
    @app.post("/api/ga/optimize")
    def ga_run():
        payload=request.get_json(silent=True) or {}
        try: result=optimize(population=bounded(payload,'population',24,4,100),generations=bounded(payload,'generations',12,1,100),inversion=bool(payload.get('inversion',False)),deletion=bool(payload.get('deletion',False)))
        except (ValueError,FileNotFoundError) as exc: return jsonify(error=f"GA: {exc}"),400
        state['partition']=decode(result['best'])
        insert_run('GeneticAlgorithmRuns',result); return jsonify(result)
    @app.post('/api/anfis/train')
    def anfis_train():
        payload=request.get_json(silent=True) or {}; samples=payload.get('samples',[])
        if not isinstance(samples,list) or len(samples)>5000: return jsonify(error='samples must be a list of at most 5000 records'),400
        try:
            X=[[float(s[k]) for k in ('contamination','moisture','recyclability','hazard')] for s in samples]
            y=[float(s['priority']) for s in samples]
            if len(samples)<10 or any(not 0<=v<=100 for row in X for v in row) or any(not 0<=v<=100 for v in y): raise ValueError('Provide at least 10 labeled samples; inputs and priority targets must be 0–100.')
            import numpy as np
            order=np.random.default_rng(42).permutation(len(X)); ntrain=max(6,int(.7*len(X))); nval=max(1,int(.15*len(X)))
            train_ix=order[:ntrain]; val_ix=order[ntrain:ntrain+nval]; test_ix=order[ntrain+nval:]
            xa=np.asarray(X); ya=np.asarray(y); model=SugenoANFIS(state['partition'])
            history=model.fit(xa[train_ix],ya[train_ix],validation=(xa[val_ix],ya[val_ix]),epochs=bounded(payload,'epochs',20,1,200))
            state['anfis']=model; state['anfis_metrics']=model.metrics(xa[test_ix],ya[test_ix])
            return jsonify(status='trained in memory for this server process',metrics=state['anfis_metrics'],history=history,training_samples=len(train_ix),validation_samples=len(val_ix),test_samples=len(test_ix),target='user-supplied priority scores',split='fixed-seed 70/15/15 approximate split')
        except (KeyError,TypeError,ValueError) as exc: return jsonify(error=str(exc)),400
    @app.get('/api/anfis/status')
    def anfis_status(): return jsonify(status='trained' if state['anfis'] else 'untrained',metrics=state['anfis_metrics'])
    @app.get("/api/export.csv")
    def export_csv():
        rows=recent(10000); out=io.StringIO(); fields=list(rows[0]) if rows else ['id','predicted_category','suitability','priority','created_at']; w=csv.DictWriter(out,fieldnames=fields); w.writeheader(); w.writerows(rows)
        return Response(out.getvalue(),mimetype='text/csv',headers={'Content-Disposition':'attachment; filename=ecosort-analyses.csv'})
    @app.get("/api/report.pdf")
    def report_pdf():
        rows=recent(10000)
        lines=['EcoSort Analysis Report',f"Generated report | {len(rows)} stored analyses | EcoSort {MODEL_VERSION}",'']
        lines += [f"#{row['id']} {row['corrected_category'] or row['predicted_category'] or 'Unclassified'} | suitability {row['suitability']:.1f} | priority {row['priority']:.1f} | {row['created_at']}" for row in rows[:35]]
        # Small dependency-free, single-page PDF; non-ASCII labels are safely transliterated.
        commands=['BT','/F1 11 Tf','50 750 Td','14 TL']
        for line in lines[:48]:
            safe=line.encode('ascii','replace').decode('ascii').replace('\\','\\\\').replace('(','\\(').replace(')','\\)')
            commands.append(f'({safe}) Tj T*')
        commands.append('ET'); stream='\n'.join(commands).encode('ascii')
        objects=[b'<< /Type /Catalog /Pages 2 0 R >>',b'<< /Type /Pages /Kids [3 0 R] /Count 1 >>',b'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>',b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>',b'<< /Length '+str(len(stream)).encode()+b' >>\nstream\n'+stream+b'\nendstream']
        pdf=bytearray(b'%PDF-1.4\n% EcoSort\n'); offsets=[0]
        for i,obj in enumerate(objects,1): offsets.append(len(pdf)); pdf.extend(f'{i} 0 obj\n'.encode()+obj+b'\nendobj\n')
        xref=len(pdf); pdf.extend(f'xref\n0 {len(objects)+1}\n0000000000 65535 f \n'.encode())
        for off in offsets[1:]: pdf.extend(f'{off:010d} 00000 n \n'.encode())
        pdf.extend(f'trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF'.encode())
        return Response(bytes(pdf),mimetype='application/pdf',headers={'Content-Disposition':'attachment; filename=ecosort-report.pdf'})
    @app.get("/")
    def index(): return send_from_directory(app.static_folder,"index.html")
    @app.get("/<path:page>")
    def frontend_page(page):
        target=Path(app.static_folder)/page
        if target.is_file(): return send_from_directory(app.static_folder,page)
        return send_from_directory(app.static_folder,"index.html")
    return app

if __name__=="__main__": create_app().run(debug=True)
