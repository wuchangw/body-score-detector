"""Offline local image-scoring UI; serves only on loopback by default."""
import argparse
import base64
import io
import json
import mimetypes
import os
import threading
import time
import traceback
import uuid
import urllib.parse
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from model import load_checkpoint, load_image, format_dimension_result

ROOT=Path(__file__).resolve().parent
EXAMPLES=ROOT.parent/'test'
LOCK=threading.Lock()
MODEL=None
CHECKPOINT=None

def font(size):
    for name in ['C:/Windows/Fonts/msyh.ttc','C:/Windows/Fonts/simhei.ttf']:
        if Path(name).exists():
            return ImageFont.truetype(name,size)
    return ImageFont.load_default()

def report(image,result):
    canvas=Image.new('RGB',(1280,840),'#F6F7FB')
    draw=ImageDraw.Draw(canvas)
    draw.text((40,26),'照片评分报告',fill='#253347',font=font(30))
    draw.text((40,74),'腹肌口径' if result['abdomen_mode']=='abs' else '马甲线口径',fill='#70798D',font=font(18))
    picture=ImageOps.contain(image,(550,640))
    canvas.paste(picture,(40+(550-picture.width)//2,120))
    draw.rounded_rectangle((630,120,1240,760),radius=24,fill='white')
    total=result['total_score']
    draw.text((665,145),'综合分',fill='#70798D',font=font(24))
    draw.text((665,184),str(total) if total!='NA' else 'NA',fill='#6554C0',font=font(65))
    draw.text((665,275),'总分为可评项目的平均值，NA 不计入',fill='#70798D',font=font(17))
    items=[('腹部评分','abdomen_score','#7968CE'),('相对肩宽','shoulder_score','#56A6BA'),('手臂维度','arm_score','#D5A467')]
    for i,(label,key,color) in enumerate(items):
        y=342+i*110
        value=result[key]
        draw.text((665,y),label,fill='#253347',font=font(22))
        draw.text((1090,y),f'{value:.1f}' if value!='NA' else 'NA',fill=color if value!='NA' else '#9299A5',font=font(25))
        draw.rounded_rectangle((665,y+45,1198,y+60),radius=7,fill='#EEF0F5')
        if value!='NA' and value>0:
            draw.rounded_rectangle((665,y+45,665+int(value*5.33),y+60),radius=7,fill=color)
    draw.text((40,790),'8 张样本训练的功能版；评分及可评性判断尚未验证准确性。',fill='#8A7655',font=font(18))
    return canvas

class Handler(BaseHTTPRequestHandler):
    def reply(self,code,data,mime='application/json; charset=utf-8'):
        if not isinstance(data,bytes):
            data=json.dumps(data,ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type',mime)
        self.send_header('Content-Length',str(len(data)))
        self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Cache-Control','no-store')
        self.end_headers();self.wfile.write(data)

    def do_GET(self):
        route=urllib.parse.urlsplit(self.path).path
        if route=='/':
            return self.reply(200,(ROOT/'index.html').read_bytes(),'text/html; charset=utf-8')
        if route=='/health':
            return self.reply(200,{'app':'boji-score-local-v2','ready':MODEL is not None,'device':'cpu'})
        if route=='/api/examples':
            files=[p for p in sorted(EXAMPLES.glob('*')) if p.is_file() and p.suffix.lower() in ('.jpg','.jpeg','.png','.jfif','.webp')][:12]
            return self.reply(200,[{'name':p.name,'url':'/examples/'+urllib.parse.quote(p.name)} for p in files])
        if route.startswith('/examples/'):
            name=urllib.parse.unquote(route[len('/examples/'):])
            if name!=Path(name).name or '\\' in name or '/' in name:
                return self.reply(404,{'error':'未找到照片'})
            file=EXAMPLES/name
            if file.is_file() and file.suffix.lower() in ('.jpg','.jpeg','.png','.jfif','.webp'):
                return self.reply(200,file.read_bytes(),mimetypes.guess_type(name)[0] or 'image/jpeg')
        if route.startswith('/results/'):
            parts=route.split('/')
            if len(parts)==4 and len(parts[2])==32 and all(c in '0123456789abcdef' for c in parts[2]) and parts[3] in ('result.json','report.png'):
                file=ROOT/'results'/parts[2]/parts[3]
                if file.is_file():
                    return self.reply(200,file.read_bytes(),'image/png' if file.suffix=='.png' else 'application/json; charset=utf-8')
        self.reply(404,{'error':'未找到内容'})

    def do_POST(self):
        if self.path!='/api/predict':
            return self.reply(404,{'error':'未找到接口'})
        try:
            size=int(self.headers.get('Content-Length','0'))
            if not 0<size<=15*1024*1024:
                return self.reply(413,{'error':'照片不能超过 10 MB'})
            request=json.loads(self.rfile.read(size))
            mode=request.get('mode','abs')
            if mode not in ('abs','lines'):
                raise ValueError('invalid mode')
            data=base64.b64decode(request['image'],validate=True)
            if len(data)>10*1024*1024:
                return self.reply(413,{'error':'照片不能超过 10 MB'})
            with Image.open(io.BytesIO(data)) as raw:
                if raw.width*raw.height>30_000_000:
                    return self.reply(413,{'error':'图片分辨率过大，请缩小后再试'})
                if min(raw.size)<64:
                    return self.reply(400,{'error':'图片太小，请使用清晰照片'})
                image=ImageOps.exif_transpose(raw).convert('RGB')
                image.thumbnail((1600,1600))
        except (ValueError,KeyError,TypeError,OSError,Image.DecompressionBombError):
            return self.reply(400,{'error':'无法读取图片，请选择 JPG、PNG 或 WebP 照片'})
        if not LOCK.acquire(blocking=False):
            return self.reply(429,{'error':'正在处理一张照片，请稍后再试'})
        try:
            run_id=uuid.uuid4().hex
            out=ROOT/'results'/run_id
            out.mkdir(parents=True)
            # Preserve original image bytes for exact parity with the server CLI.
            input_file=out/'input.image';input_file.write_bytes(data)
            start=time.perf_counter()
            with torch.inference_mode():
                output=MODEL(load_image(input_file).unsqueeze(0))
            result=format_dimension_result(output['scores'][0],output['assessability_logits'][0],mode,
                                           trained=CHECKPOINT['trained_rubrics'])
            result['filename']=str(request.get('filename','uploaded_image'))[:200]
            result['inference_ms']=round((time.perf_counter()-start)*1000)
            result['model_version']='multitask_001'
            result['rubric_note']='8-image pilot; unvalidated on new images'
            (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
            report(image,result).save(out/'report.png')
            response=dict(result,json_url=f'/results/{run_id}/result.json',report_url=f'/results/{run_id}/report.png')
            self.reply(200,response)
        except Exception:
            traceback.print_exc()
            self.reply(500,{'error':'评分失败，请查看应用日志'})
        finally:
            LOCK.release()

def main():
    global MODEL,CHECKPOINT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=7868)
    parser.add_argument('--open-browser',action='store_true')
    args=parser.parse_args()
    url=f'http://127.0.0.1:{args.port}'
    # Reuse an existing instance on double-click, without creating a second server.
    try:
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url+'/health',timeout=1) as response:
            existing=json.load(response)
        if existing.get('app')=='boji-score-local-v2':
            if args.open_browser:webbrowser.open(url)
            print('Already running: '+url,flush=True);return
    except Exception:
        pass
    torch.set_num_threads(2)
    MODEL,CHECKPOINT=load_checkpoint(ROOT/'models/model_v2.pt','cpu')
    if CHECKPOINT.get('format_version')!=2:
        raise ValueError('This UI requires the three-dimension checkpoint')
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    (ROOT/'app.pid').write_text(str(os.getpid()),encoding='ascii')
    print('Ready: '+url,flush=True)
    if args.open_browser:
        threading.Timer(0.5,lambda:webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        pid_file=ROOT/'app.pid'
        if pid_file.exists() and pid_file.read_text(encoding='ascii').strip()==str(os.getpid()):
            pid_file.unlink()

if __name__=='__main__':
    main()
