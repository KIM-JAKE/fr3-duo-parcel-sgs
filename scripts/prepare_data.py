"""Download immutable data, verify bytes, and relocate paths for this checkout."""
import argparse,pathlib,json,hashlib,shutil,xml.etree.ElementTree as ET
p=argparse.ArgumentParser();p.add_argument('--data',type=pathlib.Path,default=pathlib.Path('data'));p.add_argument('--revision',default='57561a95da538590bca3dc83d63c108f67eef43f');p.add_argument('--local-only',action='store_true');a=p.parse_args();root=pathlib.Path(__file__).resolve().parents[1];data=a.data.resolve();data.mkdir(parents=True,exist_ok=True)
if not a.local_only:
 from huggingface_hub import snapshot_download
 snapshot_download('jaeikkim/fr3-duo-parcel-resets-200k',repo_type='dataset',revision=a.revision,local_dir=data)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
checks=json.loads((data/'SHA256SUMS.json').read_text())
for item in checks['files']:
 path=data/item['path'];assert path.is_file() and path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],item['path']
idx=root/'data_index';idx.mkdir(exist_ok=True);m=json.loads((data/'index/manifest.json').read_text());m['shards']=[str(data/x) for x in m['shards']];m['dataset_revision']=a.revision;m['dataset_repo']='jaeikkim/fr3-duo-parcel-resets-200k'
for name in ['configurations.npz','ng_free_flags.npz','ng_sampling_contract.json']:shutil.copy2(data/'index'/name,idx/name)
(idx/'manifest.json').write_text(json.dumps(m,indent=2));assert sha(idx/'configurations.npz')==m['configurations_sha256']
# Local relocatable asset clone. Never rewrite the downloaded release bytes.
assets=root/'assets';shutil.copytree(data/'assets',assets,dirs_exist_ok=True);urdf=assets/'robot/fr3_duo_dg5fm.urdf';tree=ET.parse(urdf)
for elem in tree.iter('mesh'):
 path=pathlib.Path(elem.attrib['filename']);path=path if path.is_absolute() else urdf.parent/path;assert path.is_file();elem.set('filename',str(path.resolve()))
tree.write(urdf,encoding='utf-8',xml_declaration=True)
cell=root/'fr3_duo_sgs/cell_config.json';c=json.loads(cell.read_text());c['robot']['urdf']=str(urdf);c['robot']['sha256']=sha(urdf);cell.write_text(json.dumps(c,indent=2))
usd_root=root/'fr3_duo_sgs/runtime/native_duo_usd';shutil.copytree(assets/'native_duo_usd',usd_root,dirs_exist_ok=True);audit=json.loads((usd_root/'usd_audit.json').read_text())
for candidate in audit['candidates']:
 old=pathlib.Path(candidate['usd']);parts=old.parts;pos=parts.index('native_duo_usd');candidate['usd']=str(usd_root/pathlib.Path(*parts[pos+1:]))
(usd_root/'usd_audit.json').write_text(json.dumps(audit,indent=2));(root/'prepared_data.json').write_text(json.dumps(dict(data=str(data),index=str(idx),revision=a.revision,configuration_count=m['configuration_count'],verified_files=len(checks['files'])),indent=2));print('Verified and prepared',m['configuration_count'],'resets at',idx)
