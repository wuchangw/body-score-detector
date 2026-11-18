"""Input one image; output one appearance score between zero and 100."""
import argparse
import json
from pathlib import Path
import torch
from model import load_checkpoint,load_image,format_dimension_result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image',required=True)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--output',help='Optional output JSON path')
    parser.add_argument('--abdomen-mode',choices=['abs','lines'],default='abs',
                        help='abs: male abdomen rubric; lines: female waist-line rubric (user-selected)')
    parser.add_argument('--visibility-threshold',type=float,default=0.5)
    args=parser.parse_args()
    torch.set_num_threads(2)
    model,checkpoint=load_checkpoint(args.checkpoint,args.device)
    with torch.inference_mode():
        value=model(load_image(args.image).unsqueeze(0).to(args.device))
    if checkpoint['format_version']==2:
        result=format_dimension_result(value['scores'][0],value['assessability_logits'][0],
                                       args.abdomen_mode,args.visibility_threshold,
                                       checkpoint['trained_rubrics'])
    else:
        value=float(value.item())
        if not 0<=value<=100:
            raise ValueError('Invalid predicted score')
        result={'score':round(value,1)}
    if args.output:
        Path(args.output).parent.mkdir(parents=True,exist_ok=True)
        Path(args.output).write_text(json.dumps(result),encoding='utf-8')
    print(json.dumps(result))

if __name__=='__main__':
    main()
