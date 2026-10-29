"""Restore selected model paths in this checkout; never overwrite different files."""
import hashlib,json,shutil
from pathlib import Path
root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'models/manifest.json').read_text())
for m in manifest['models']:
    src=root/m['path'];dst=root/m['restore_path']
    assert hashlib.sha256(src.read_bytes()).hexdigest()==m['sha256'],src
    if dst.exists():
        assert hashlib.sha256(dst.read_bytes()).hexdigest()==m['sha256'], 'Different existing file: '+str(dst)
    else:
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    print(dst.relative_to(root))
