"""Train masked rubric scores and assessability on the existing small pilot."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from model import DimensionScoreModel,load_image,load_checkpoint,masked_score_loss,format_dimension_result

def tensors(records,device):
    targets=[];masks=[];observed=[]
    for item in records:
        scores=item['scores']
        visible=[scores[k] is not None for k in ('abdomen','shoulder','arm')]
        references=item.get('abdomen_references',{})
        raw=[references.get('abs'),references.get('lines'),scores['shoulder'],scores['arm']]
        if not visible[0]:
            raw[0]=raw[1]=None
        for value in raw:
            if value is not None and (isinstance(value,bool) or not isinstance(value,(int,float)) or not 0<=value<=100):
                raise ValueError('Invalid component label')
        targets.append([value if value is not None else 0.0 for value in raw])
        masks.append([value is not None for value in raw])
        observed.append(visible)
    return (torch.tensor(targets,dtype=torch.float32,device=device),
            torch.tensor(masks,dtype=torch.bool,device=device),
            torch.tensor(observed,dtype=torch.float32,device=device))

def run(args):
    torch.set_num_threads(2);torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    path=Path(args.manifest).resolve();root=path.parent
    manifest=json.loads(path.read_text(encoding='utf-8'))
    records=[];seen={}
    for item in manifest['records']:
        file=(root/item['image']).resolve();file.relative_to(root)
        digest=hashlib.sha256(file.read_bytes()).hexdigest()
        if digest!=item['sha256']:
            raise ValueError('Image checksum mismatch')
        if digest in seen:
            if any(item.get(k)!=seen[digest].get(k) for k in ('scores','abdomen_references')):
                raise ValueError('Conflicting duplicate labels')
            continue
        seen[digest]=item;records.append(dict(item,path=file))
    if len(records)<2:
        raise ValueError('At least two unique images required')
    out=Path(args.output)
    if (out/'model.pt').exists():
        raise FileExistsError('Choose a new run directory')
    out.mkdir(parents=True,exist_ok=True)
    device=args.device
    images=torch.stack([load_image(r['path']) for r in records]).to(device)
    targets,mask,observed=tensors(records,device)
    if not mask.any():
        raise ValueError('No valid component labels')
    model=DimensionScoreModel().to(device)
    weights=torch.load(args.pretrained,map_location='cpu',weights_only=True)
    model.encoder.load_state_dict({k:v for k,v in weights.items() if not k.startswith('fc.')},strict=True)
    model.encoder.eval()
    with torch.no_grad():
        features=model.encode(images)
        model.feature_mean.copy_(features.mean(0))
        model.feature_scale.copy_(features.std(0,unbiased=False).clamp_min(0.1))
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.Adam(params,lr=0.002)
    history=[];best=float('inf');best_state=None;best_step=0
    for step in range(args.steps+1):
        result=model.score_features(features)
        regression=masked_score_loss(result['scores'],targets,mask)
        assessability=F.binary_cross_entropy_with_logits(result['assessability_logits'],observed)
        loss=regression+0.1*assessability
        with torch.no_grad():
            errors=(result['scores']-targets).abs()[mask]
            mae=float(errors.mean());max_error=float(errors.max())
            probabilities=result['assessability_logits'].sigmoid()
            accuracy=float(((probabilities>=0.5)==observed.bool()).float().mean())
            confidence=float(torch.where(observed.bool(),probabilities,1-probabilities).min())
        passed=mae<=0.2 and max_error<=0.5 and accuracy==1.0 and confidence>=0.8
        if float(loss.detach())<best or passed:
            best=float(loss.detach());best_step=step
            best_state=copy.deepcopy({k:v.detach().cpu() for k,v in model.state_dict().items()})
        row={'step':step,'loss':float(loss.detach()),'valid_label_mae':mae,
             'valid_label_max_error':max_error,'assessability_accuracy':accuracy}
        history.append(row)
        if step%100==0 or passed:
            print(json.dumps(row),flush=True)
        if step==args.steps or passed:
            break
        optimizer.zero_grad();loss.backward();optimizer.step()
    trained=mask.any(0).tolist()
    checkpoint={'format_version':2,'architecture':'frozen_resnet18_mlp64_rubrics4_assessability3',
                'model_state':best_state,'trained_rubrics':trained,
                'score_order':['abdomen_abs','abdomen_lines','shoulder','arm'],
                'assessability_order':['abdomen','shoulder','arm'],'image_size':224,
                'preprocess':'EXIF RGB aspect-preserving pad128 ImageNet normalization',
                'training_mode':'pilot_overfit_no_holdout','best_step':best_step,'seed':args.seed,
                'training_hashes':[r['sha256'] for r in records],
                'label_source':manifest.get('label_source',''),
                'abdomen_mode_selection':'user-specified; no gender inference',
                'total_rule':'equal mean of assessable dimensions; NA omitted'}
    torch.save(checkpoint,out/'model.pt')
    restored,_=load_checkpoint(out/'model.pt',device)
    with torch.inference_mode():
        result=restored(images)
        errors=(result['scores']-targets).abs()
        predicted=(result['assessability_logits'].sigmoid()>=0.5)
    accuracy=float((predicted==observed.bool()).float().mean())
    mae=float(errors[mask].mean());max_error=float(errors[mask].max())
    predictions=[]
    for i,record in enumerate(records):
        mode='lines' if record['abdomen_mode']=='马甲线' else 'abs'
        formatted=format_dimension_result(result['scores'][i],result['assessability_logits'][i],mode,trained=trained)
        predictions.append({'image':record['image'],'targets':record['scores'],'prediction':formatted})
    rubric_metrics={}
    for j,key in enumerate(checkpoint['score_order']):
        selected=mask[:,j]
        rubric_metrics[key]={'labels':int(selected.sum()),'train_mae':float(errors[:,j][selected].mean()) if selected.any() else None}
    summary={'training_mode':'pilot_overfit_no_holdout','unique_images':len(records),
             'best_step':best_step,'train_valid_label_mae':mae,'train_max_error':max_error,
             'train_assessability_accuracy':accuracy,'rubrics':rubric_metrics,
             'component_label_counts':dict(zip(('abdomen','shoulder','arm'),observed.sum(0).int().tolist())),
             'generalization_verified':False,'heldout_evaluation':None,
             'pipeline_passed':max_error<=0.5 and accuracy==1.0,'checkpoint_reload_verified':True,
             'seed':args.seed,'device':device,'torch_version':str(torch.__version__)}
    for name,data in [('metrics.json',summary),('history.json',history),('train_predictions.json',predictions)]:
        (out/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)
    if not summary['pipeline_passed']:
        raise SystemExit('Training ran but the pilot fitting check did not pass')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--pretrained',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--steps',type=int,default=2500)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--device',default='cpu')
    run(parser.parse_args())
