"""Build public teaching catalog from development fixture; not scored evaluation data."""
from pathlib import Path
import json, hashlib, shutil
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
fixture=ROOT/'data/processed/galaxy-zoo-2/fixture'
labels=json.loads((fixture/'labels.private.json').read_text())
records=json.loads((fixture/'objects.json').read_text())
out=ROOT/'apps/web/client/public/galaxies';out.mkdir(parents=True,exist_ok=True)
def count(r,key):return int(float(r.get(key+'_count',0)))
def descriptor(im):
 im=im.convert('RGB').resize((16,16),Image.Resampling.BILINEAR)
 vals=[(r*.299+g*.587+b*.114)/255 for r,g,b in im.getdata()]
 mean=sum(vals)/len(vals);norm=sum((v-mean)**2 for v in vals)**.5 or 1
 return [round((v-mean)/norm,5) for v in vals]
catalog=[]
for obj in records:
 r=labels[obj['object_id']];p=fixture/obj['image'];b=p.read_bytes()
 smooth=count(r,'t01_smooth_or_features_a01_smooth');features=count(r,'t01_smooth_or_features_a02_features_or_disk');total=smooth+features+count(r,'t01_smooth_or_features_a03_star_or_artifact')
 edge=count(r,'t02_edgeon_a04_yes');noedge=count(r,'t02_edgeon_a05_no');spiral=count(r,'t04_spiral_a08_spiral');nospiral=count(r,'t04_spiral_a09_no_spiral')
 kind='Uncertain structure';desc='The classifications do not establish a single clear structure. A closer look or a sharper image could help.'
 if total and smooth/total>=.8:kind='Smooth appearance';desc='Most volunteers described a smooth-looking galaxy. Fine structure is not established by that classification.'
 elif total and features/total>=.6:
  kind='Visible features';desc='Most volunteers identified features or a disk rather than a simply smooth appearance.'
  if edge+noedge>=10 and edge/(edge+noedge)>=.8:kind='Edge-on appearance';desc='The votes favor a disk seen from the side. This orientation can make its overall structure harder to distinguish.'
  elif noedge>=10 and spiral+nospiral>=10 and spiral/(spiral+nospiral)>=.8:kind='Spiral structure';desc='The volunteers who reached the spiral question strongly favored visible spiral arms.'
 im=Image.open(p);im.load();descvec=descriptor(im);shutil.copyfile(p,out/p.name)
 catalog.append({'id':obj['object_id'],'assetId':obj['asset_id'],'name':'SDSS '+obj['object_id'],'image':'/galaxies/'+p.name,'ra':float(obj['ra']),'dec':float(obj['dec']),'kind':kind,'description':desc,'votes':{'smooth':smooth,'features':features,'total':total,'spiral':spiral,'noSpiral':nospiral},'hash':hashlib.sha256(b).hexdigest(),'descriptor':descvec})
catalog.sort(key=lambda x:(x['kind']!='Spiral structure',x['kind']!='Edge-on appearance',x['id']))
(ROOT/'apps/web/client/public/catalog.json').write_text(json.dumps(catalog,separators=(',',':')))
print('Built catalog:',len(catalog),'objects; all images decoded.')
