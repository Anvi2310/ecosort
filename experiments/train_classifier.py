"""Fine-tune MobileNetV2 on the documented six-folder EcoSort dataset."""
import argparse
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from config import CATEGORIES, MODEL_DIR, ROOT

def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--data',type=Path,default=ROOT/'data/processed/images'); parser.add_argument('--epochs',type=int,default=8); parser.add_argument('--batch-size',type=int,default=24); parser.add_argument('--seed',type=int,default=41); args=parser.parse_args()
    torch.manual_seed(args.seed)
    means=[.485,.456,.406]; stds=[.229,.224,.225]
    train_tf=transforms.Compose([transforms.Resize((232,232)),transforms.RandomResizedCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(means,stds)])
    eval_tf=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize(means,stds)])
    train=datasets.ImageFolder(args.data/'train',transform=train_tf); val=datasets.ImageFolder(args.data/'val',transform=eval_tf); test=datasets.ImageFolder(args.data/'test',transform=eval_tf)
    expected=sorted(x.lower().replace(' ','_') for x in CATEGORIES)
    if train.classes!=expected or val.classes!=expected or test.classes!=expected: raise SystemExit(f"All splits must use exactly these sorted class folders: {expected}; got {train.classes}")
    if not train or not val or not test: raise SystemExit('Train, validation and test splits must all contain images.')
    loaders={k:DataLoader(v,batch_size=args.batch_size,shuffle=(k=='train'),num_workers=0) for k,v in [('train',train),('val',val),('test',test)]}
    model=models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    for p in model.features.parameters(): p.requires_grad=False
    model.classifier[1]=nn.Linear(model.last_channel,len(expected)); model=model.to('cuda' if torch.cuda.is_available() else 'cpu'); device=next(model.parameters()).device
    opt=torch.optim.AdamW(model.classifier.parameters(),lr=1e-3); loss_fn=nn.CrossEntropyLoss(); best=float('inf'); best_state=None
    for epoch in range(args.epochs):
        model.train()
        for x,y in loaders['train']:
            x,y=x.to(device),y.to(device); opt.zero_grad(); loss=loss_fn(model(x),y); loss.backward(); opt.step()
        model.eval(); losses=[]
        with torch.no_grad():
            for x,y in loaders['val']: losses.append(loss_fn(model(x.to(device)),y.to(device)).item())
        val_loss=sum(losses)/max(1,len(losses)); print(f"epoch={epoch+1} validation_loss={val_loss:.5f}")
        if val_loss<best: best=val_loss; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state); model.eval(); truth=[]; pred=[]
    with torch.no_grad():
        for x,y in loaders['test']: pred.extend(model(x.to(device)).argmax(1).cpu().tolist()); truth.extend(y.tolist())
    print('Test accuracy:',accuracy_score(truth,pred)); print(classification_report(truth,pred,target_names=expected,zero_division=0)); print('Confusion matrix:\n',confusion_matrix(truth,pred))
    MODEL_DIR.mkdir(parents=True,exist_ok=True); path=MODEL_DIR/'waste_mobilenet_v2.pth'; torch.save(model.cpu().state_dict(),path)
    print(f'Saved fine-tuned weights: {path}. Test results above are from this dataset split.')

if __name__=='__main__': main()
