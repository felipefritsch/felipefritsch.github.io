from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlparse,unquote
import re,json
root=Path(__file__).resolve().parents[1]/'docs'
class Page(HTMLParser):
 def __init__(self):super().__init__();self.ids=set();self.urls=[]
 def handle_starttag(self,t,a):
  for k,v in a:
   if k=='id':self.ids.add(v)
   if k in ('href','src'):self.urls.append(v)
pages={}
for p in root.rglob('*.html'):
 x=Page();text=p.read_text();x.feed(text);pages[p.resolve()]=x
 assert not any(v in text for v in ['/Users/','file:///','Q25','Jordan Brooks','My notes for Part']),p
errors=[];count=0
for p,page in pages.items():
 for raw in page.urls:
  u=urlparse(raw)
  if u.scheme:continue
  target=(p.parent/unquote(u.path)).resolve() if u.path else p
  if target.is_dir():target=target/'index.html'
  count+=1
  if not target.exists():errors.append((str(p),raw,'missing file'))
  elif u.fragment and target in pages and not u.fragment.startswith(':~:') and unquote(u.fragment) not in pages[target].ids:errors.append((str(p),raw,'missing anchor'))
assert not errors,errors
for name in ['Felipe_Fritsch_CV.pdf','Felipe_Fritsch_Columbia_Thesis.pdf']:
 assert (root/name).read_bytes().startswith(b'%PDF'),name
print(json.dumps({'html_pages':len(pages),'local_links_checked':count,'errors':errors,'pdf_downloads':2}))
