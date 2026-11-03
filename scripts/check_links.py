#!/usr/bin/env python3
"""Check repository Markdown relative destinations without following local evidence paths."""
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote

root=Path(__file__).resolve().parents[1]
tracked=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','*.md'],cwd=root,text=True).splitlines()
errors=[]
for name in tracked:
    path=root/name
    text=re.sub(r'```.*?```','',path.read_text(),flags=re.S)
    for target in re.findall(r'\]\(<?([^\s)>]+)>?(?:\s+[^)]*)?\)',text):
        target=unquote(target.split('#',1)[0])
        if not target or '://' in target or target.startswith('mailto:'):continue
        destination=(path.parent/target).resolve()
        if not destination.is_relative_to(root) or not destination.exists():
            errors.append(f'{name}: {target}')
if errors:raise SystemExit('\n'.join(errors))
print(f'All relative links valid in {len(tracked)} Markdown documents')
