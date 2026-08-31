from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] if Path(__file__).name != 'apply_browser_quality_fix.py' else Path.cwd()


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f'{label}: expected exactly one match, got {count}')
    return text.replace(old, new, 1)


def replace_function(text: str, start: str, next_start: str, replacement: str, label: str) -> str:
    pattern = re.escape(start) + r'.*?(?=\n' + re.escape(next_start) + r')'
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.S)
    if count != 1:
        raise RuntimeError(f'{label}: expected one function match, got {count}')
    return updated


js_path = Path('docs/browser-lab.js')
js = js_path.read_text(encoding='utf-8')

js = replace_function(
    js,
    'function autoPlan(img,a){',
    'function fft1(',
    "function autoPlan(img,a){const minDim=Math.min(img.naturalWidth,img.naturalHeight),raw=[21,31,45,65,85,105,125].filter(v=>v<Math.max(31,minDim*.28));let supports=raw.length>=4?raw:[21,31,45,65,85].filter(v=>v<minDim*.32);if(a.edge>.085&&a.hp>.14)supports=supports.slice(0,4);else if(a.edge<.03)supports=supports.slice(-4);else if(supports.length>5)supports=supports.slice(1,6);supports=[...new Set(supports.map(odd))];if(supports.length<3)supports=[15,25,35].filter(v=>v<minDim*.35);const gammas=(a.lowContrast||a.brightFrac>.08||a.mean>.56)?[1,2.2]:[1],modes=a.lowLight||a.highSaturation?['gradient','dark']:['dark','gradient'],estimationMax=Math.min(640,Math.max(384,Math.round(Math.sqrt(minDim)*18))),pixels=img.naturalWidth*img.naturalHeight,restoreMax=pixels<=4200000?Math.max(img.naturalWidth,img.naturalHeight):2200;return{supports,gammas,modes,estimationMax,restoreMax,coarseIter:2,fineIter:5}}",
    'autoPlan',
)

js = replace_function(
    js,
    'function refineKernel(k,size,aggressive=false){',
    'function kernelStats(',
    "function refineKernel(k,size,aggressive=false){let m=0;for(const v of k)m=Math.max(m,v);const thr=m*(aggressive?.025:.015),o=new Float32Array(k.length);for(let i=0;i<k.length;i++)if(k[i]>=thr)o[i]=k[i];normalizeKernel(o);const c=(size-1)/2;let sxx=0,syy=0,sxy=0;for(let y=0;y<size;y++)for(let x=0;x<size;x++){const v=o[y*size+x],dx=x-c,dy=y-c;sxx+=v*dx*dx;syy+=v*dy*dy;sxy+=v*dx*dy}const angle=.5*Math.atan2(2*sxy,sxx-syy),ca=Math.cos(angle),sa=Math.sin(angle);if(aggressive)for(let y=0;y<size;y++)for(let x=0;x<size;x++){const i=y*size+x,v=o[i],dx=x-c,dy=y-c,along=Math.abs(dx*ca+dy*sa),off=Math.abs(-dx*sa+dy*ca);if(v&&off>Math.max(3,size*.11)&&along>size*.20&&v<m*.08)o[i]=0}return centerKernel(normalizeKernel(o),size)}",
    'refineKernel',
)

js = replace_function(
    js,
    'function latentStep(src,w,h,k,ks,useDark,lambdaDark,lambdaGrad,patch){',
    'function estimatePsfFromGradients(',
    "function latentStep(src,w,h,k,ks,useDark,lambdaDark,lambdaGrad,patch){let s=deconvChannel(src,w,h,k,ks,Math.max(lambdaGrad,.00055));if(useDark&&lambdaDark>0){let beta=Math.max(.03,lambdaDark/.03);for(let t=0;t<3;t++){const u=localMinProjection(s,w,h,patch,lambdaDark,beta);s=deconvChannel(src,w,h,k,ks,Math.max(lambdaGrad,.00055),u,beta);beta*=2}}return s}",
    'latentStep',
)

js = replace_function(
    js,
    'async function blindCandidate(base,fullSupport,mode,gamma,iterations){',
    'async function autoEstimate(',
    "async function blindCandidate(base,fullSupport,mode,gamma,iterations){const scaledSupport=odd(Math.max(7,Math.min(81,Math.round(fullSupport*base.scale)))),gbase=new Float32Array(base.gray.length);for(let i=0;i<gbase.length;i++)gbase[i]=Math.pow(clamp(base.gray[i]),gamma);const ratio=Math.SQRT1_2,maxLevels=iterations>=5?5:3,scaleLevels=[1];let sc=1;while(scaleLevels.length<maxLevels&&scaledSupport*sc>9){sc*=ratio;scaleLevels.unshift(sc)}let k=null,ks=0,lambdaD=mode==='dark'?.004:0,lambdaG=.004;for(const level of scaleLevels){const w=Math.max(40,Math.round(base.w*level)),h=Math.max(40,Math.round(base.h*level)),y=resizeGray(gbase,base.w,base.h,w,h),target=odd(Math.max(5,Math.round(scaledSupport*level)));k=k?resizeKernel(k,ks,target):makeInitKernel(target);ks=target;const bg=gradients(y,w,h);for(let it=0;it<iterations;it++){const patch=odd(Math.max(9,Math.min(35,Math.round(35*level)))),latent=latentStep(y,w,h,k,ks,mode==='dark',lambdaD,lambdaG,patch),lg=gradients(latent,w,h),tg=thresholdGrad(lg.gx,lg.gy,.08);k=estimatePsfFromGradients(bg.gx,bg.gy,tg.gx,tg.gy,w,h,ks);lambdaD=lambdaD?Math.max(.0001,lambdaD/1.1):0;lambdaG=Math.max(.0001,lambdaG/1.1)}k=centerKernel(k,ks);await nextFrame()}k=resizeKernel(k,ks,scaledSupport);k=refineKernel(k,scaledSupport,true);return{k,size:scaledSupport,fullSupport,mode,gamma,score:scoreKernel(base.gray,base.w,base.h,k,scaledSupport,fullSupport)}}",
    'blindCandidate',
)

js = replace_once(
    js,
    'const fineBase=imageArrays(S.image,Math.min(480,plan.estimationMax+110))',
    'const fineBase=imageArrays(S.image,Math.min(640,plan.estimationMax+120))',
    'fine estimation resolution',
)

js = replace_function(
    js,
    'function chooseBaseline(data,k,ks,a){',
    'function pnpRefine(',
    "function chooseBaseline(data,k,ks,a){const noise=Math.max(a.hp,.01),baseReg=clamp(.0018+noise*.014+(a.lowLight?.0015:0),.0008,.0065),regs=[baseReg*.5,baseReg*.8,baseReg,baseReg*1.5,baseReg*2.2],items=[];for(const reg of regs){const raw=restoreRGB(data.rgb,data.w,data.h,k,ks,reg),guarded=safeBlend(data.rgb,raw,data.w,data.h,k,ks);items.push({img:guarded.img,reg,q:guarded.q})}items.sort((x,y)=>x.q.score-y.q.score);return items[0]}",
    'chooseBaseline',
)

helpers = """function smoothGray(src,w,h){const tmp=new Float32Array(src.length),out=new Float32Array(src.length),a=[.25,.5,.25];for(let y=0;y<h;y++)for(let x=0;x<w;x++){let v=0;for(let j=-1;j<=1;j++)v+=src[y*w+reflect(x+j,w)]*a[j+1];tmp[y*w+x]=v}for(let y=0;y<h;y++)for(let x=0;x<w;x++){let v=0;for(let j=-1;j<=1;j++)v+=tmp[reflect(y+j,h)*w+x]*a[j+1];out[y*w+x]=v}return out}
function localGradientMap(gray,w,h){const out=new Float32Array(gray.length);for(let y=0;y<h;y++)for(let x=0;x<w;x++){const i=y*w+x,gx=gray[y*w+Math.min(w-1,x+1)]-gray[y*w+Math.max(0,x-1)],gy=gray[Math.min(h-1,y+1)*w+x]-gray[Math.max(0,y-1)*w+x];out[i]=Math.hypot(gx,gy)}return out}
function localHighpassMap(gray,w,h){const out=new Float32Array(gray.length);for(let y=0;y<h;y++)for(let x=0;x<w;x++){const i=y*w+x,l=gray[y*w+Math.max(0,x-1)],r=gray[y*w+Math.min(w-1,x+1)],u=gray[Math.max(0,y-1)*w+x],d=gray[Math.min(h-1,y+1)*w+x];out[i]=Math.abs(4*gray[i]-l-r-u-d)}return out}
"""
if 'function smoothGray(' not in js:
    js = replace_once(js, 'function rgacRefine(', helpers + 'function rgacRefine(', 'RGAC helpers')

js = replace_function(
    js,
    'function rgacRefine(obs,candidates,w,h,k,ks){',
    'async function restoreFamily(',
    "function rgacRefine(obs,candidates,w,h,k,ks){const entries=candidates.map(x=>({img:x.img,q:x.q})),observedGray=rgbToGray(obs),observedEdge=smoothGray(localGradientMap(observedGray,w,h),w,h),observedHp=smoothGray(localHighpassMap(observedGray,w,h),w,h),energies=[],globalMin=Math.min(...entries.map(x=>x.q.score)),globalScale=Math.max(Math.abs(globalMin),.01);for(const entry of entries){const gray=rgbToGray(entry.img),reblur=blurChannel(gray,w,h,k,ks),residual=new Float32Array(gray.length),edge=smoothGray(localGradientMap(gray,w,h),w,h),hp=smoothGray(localHighpassMap(gray,w,h),w,h),energy=new Float32Array(gray.length),globalEnergy=clamp((entry.q.score-globalMin)/globalScale,0,4);for(let i=0;i<gray.length;i++){residual[i]=Math.abs(reblur[i]-observedGray[i])}const smoothResidual=smoothGray(residual,w,h);for(let i=0;i<gray.length;i++){const edgeRatio=(edge[i]+.004)/(observedEdge[i]+.004),hpRatio=(hp[i]+.003)/(observedHp[i]+.003),edgePenalty=clamp((edgeRatio-1.5)/1.25,0,3),hpPenalty=clamp((hpRatio-1.65)/1.35,0,3),clipPenalty=(entry.img[i*3]<=.003||entry.img[i*3]>=.997||entry.img[i*3+1]<=.003||entry.img[i*3+1]>=.997||entry.img[i*3+2]<=.003||entry.img[i*3+2]>=.997)?.35:0;energy[i]=1.75*smoothResidual[i]/Math.max(.004,smoothResidual[i]+.012)+.8*edgePenalty+hpPenalty+.7*clipPenalty+.22*globalEnergy}energies.push(energy)}const weights=energies.map(()=>new Float32Array(w*h)),fused=new Float32Array(obs.length);for(let i=0;i<w*h;i++){let minE=Infinity;for(const e of energies)minE=Math.min(minE,e[i]);let sum=0;for(let j=0;j<energies.length;j++){const weight=Math.exp(-(energies[j][i]-minE)/.5);weights[j][i]=weight;sum+=weight}sum=Math.max(sum,1e-8);for(let j=0;j<weights.length;j++)weights[j][i]/=sum;for(let c=0;c<3;c++){let v=0;for(let j=0;j<entries.length;j++)v+=weights[j][i]*entries[j].img[i*3+c];fused[i*3+c]=clamp(v)}}const prior=gaussianRGB(fused,w,h),projected=restoreRGB(obs,w,h,k,ks,.0012,prior,.12),guarded=safeBlend(obs,projected,w,h,k,ks),meanWeights=weights.map(a=>{let sum=0;for(const v of a)sum+=v;return sum/a.length});return{...guarded,weights:meanWeights}}",
    'rgacRefine',
)

old_split = "function updateSplit(){const v=Number(E.beforeAfterSlider.value);E.resultImage.style.clipPath=`inset(0 ${100-v}% 0 0)`;E.splitLine.style.left=v+'%';E.splitHandle.style.left=v+'%'}E.beforeAfterSlider.addEventListener('input',updateSplit);"
new_split = """function updateSplit(){const v=clamp(Number(E.beforeAfterSlider.value),0,100);E.resultImage.style.clipPath=`inset(0 0 0 ${v}%)`;E.splitLine.style.left=v+'%';E.splitHandle.style.left=v+'%'}
function setSplitFromClientX(clientX){const rect=E.viewer.getBoundingClientRect(),ratio=clamp((clientX-rect.left)/Math.max(rect.width,1),0,1);E.beforeAfterSlider.value=String(Math.round(ratio*100));updateSplit()}
let splitDragging=false;
E.beforeAfterSlider.addEventListener('input',updateSplit);
E.splitHandle.addEventListener('pointerdown',event=>{if(E.splitHandle.classList.contains('hidden'))return;event.preventDefault();event.stopPropagation();splitDragging=true;E.splitHandle.setPointerCapture?.(event.pointerId);setSplitFromClientX(event.clientX)});
E.splitHandle.addEventListener('pointermove',event=>{if(splitDragging)setSplitFromClientX(event.clientX)});
for(const name of ['pointerup','pointercancel','lostpointercapture'])E.splitHandle.addEventListener(name,()=>{splitDragging=false});
E.viewer.addEventListener('pointerdown',event=>{if(E.beforeAfterSlider.classList.contains('hidden')||event.target===E.beforeAfterSlider||event.target===E.splitHandle)return;setSplitFromClientX(event.clientX)});"""
js = replace_once(js, old_split, new_split, 'before/after reveal')
js_path.write_text(js, encoding='utf-8')

css_path = Path('docs/browser-lab.css')
css = css_path.read_text(encoding='utf-8')
css = replace_once(css, 'user-select:none}.viewer .after{clip-path:inset(0 50% 0 0)}', 'user-select:none;pointer-events:none}.viewer .after{clip-path:inset(0 0 0 50%)}', 'before/after CSS direction')
css = replace_once(css, 'pointer-events:none}.handle{position:absolute', 'pointer-events:none;z-index:6}.handle{position:absolute', 'split line stacking')
css = replace_once(css, 'width:38px;height:38px', 'width:42px;height:42px', 'handle size')
css = replace_once(css, 'font-size:13px;color:#344054;pointer-events:none}.viewer-label', 'font-size:14px;color:#344054;pointer-events:auto;cursor:ew-resize;touch-action:none;z-index:7}.viewer-label', 'handle interaction')
css_path.write_text(css, encoding='utf-8')

html_path = Path('docs/index.html')
html = html_path.read_text(encoding='utf-8')
html = replace_once(html, 'Drag the slider to reveal more or less of the selected result.', 'Drag the center handle, click the image, or use the bottom slider to compare Before and After.', 'viewer hint')
html_path.write_text(html, encoding='utf-8')

test_path = Path('tests/test_report_assets.py')
test = test_path.read_text(encoding='utf-8')
anchor = '\n\ndef test_benchmark_profiles_cover_every_source_with_valid_support() -> None:\n'
addition = '''\n\ndef test_browser_before_after_reveal_is_directionally_correct_and_draggable() -> None:\n    page = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")\n    script = (ROOT / "docs" / "browser-lab.js").read_text(encoding="utf-8")\n    styles = (ROOT / "docs" / "browser-lab.css").read_text(encoding="utf-8")\n\n    assert "Drag the center handle" in page\n    assert ".viewer .after{clip-path:inset(0 0 0 50%)}" in styles\n    assert "pointer-events:auto;cursor:ew-resize;touch-action:none" in styles\n    assert "function setSplitFromClientX" in script\n    assert "E.splitHandle.addEventListener('pointerdown'" in script\n    assert "E.resultImage.style.clipPath=`inset(0 0 0 ${v}%)`" in script\n\n\ndef test_browser_quality_profile_tracks_python_pipeline_more_closely() -> None:\n    script = (ROOT / "docs" / "browser-lab.js").read_text(encoding="utf-8")\n\n    # The browser remains self-contained, but its quality path should mirror\n    # the Python pipeline's stronger blind search and residual-guided consensus.\n    assert "estimationMax=Math.min(640" in script\n    assert "Math.min(640,plan.estimationMax+120)" in script\n    assert "fineIter:5" in script\n    assert "Math.SQRT1_2" in script\n    assert "Math.min(35,Math.round(35*level))" in script\n    assert "function smoothGray" in script\n    assert "function localGradientMap" in script\n    assert "function localHighpassMap" in script\n    assert "projected=restoreRGB" in script\n'''
if 'test_browser_before_after_reveal_is_directionally_correct_and_draggable' not in test:
    test = replace_once(test, anchor, addition + anchor, 'test insertion')
test_path.write_text(test, encoding='utf-8')
