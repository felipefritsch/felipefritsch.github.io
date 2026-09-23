// Regenerate only the reading edition. Study_Wiki.md remains the editable master.
const fs = require('node:fs');
const path = require('node:path');
const {pathToFileURL} = require('node:url');
const dir = path.resolve(__dirname, '../docs/wiki');
const sourceAssets = path.resolve(dir, '../assets');
const marked = require(path.join(sourceAssets, 'marked.umd.js'));
const katex = require(path.join(sourceAssets, 'katex/katex.min.js'));
const md = fs.readFileSync(path.join(dir, 'Study_Wiki.md'), 'utf8');
const escape = s => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
const math = [];
const input = md.replace(/^\$\$\s*\n([\s\S]*?)\n\$\$\s*$/gm, (_, equation) => {
	const rendered = katex.renderToString(equation, {displayMode:true, throwOnError:true, strict:'error', trust:false});
	math.push(rendered);
	return '\nWIKIMATHBLOCK' + (math.length-1) + '\n';
});
let html = marked.parse(input);
html = html.replace(/<p>WIKIMATHBLOCK(\d+)<\/p>/g, (_, i) => '<div class="equation">'+math[Number(i)]+'</div>');
const ids = new Set([...html.matchAll(/id="([^"]+)"/g)].map(m=>m[1]));
const headings=[];
html=html.replace(/<h([1-6])>([\s\S]*?)<\/h\1>/g,(_,level,label)=>{
	const plain=label.replace(/<[^>]*>/g,'').replace(/&amp;/g,'&');
	const base=plain.toLowerCase().replace(/[^a-z0-9\s-]/g,'').trim().replace(/\s+/g,'-');
	let id=base, n=2; while(ids.has(id)) id=base+'-'+n++;
	ids.add(id);headings.push({level:Number(level),label:plain,id});
	return `<h${level} id="${id}">${label}</h${level}>`;
});
// Only local document links are converted to file URLs. External links stay external.
html=html.replace(/href="(\/Users\/[^"#]*)(#[^"]*)?"/g,(_,p,hash)=>`href="${escape(pathToFileURL(decodeURI(p)).href+(hash||''))}"`);
html=html.replace(/<table>/g,'<div class="table-wrap"><table>').replace(/<\/table>/g,'</table></div>');
// Provide a local contents list inside each long part without changing editable prose.
for(let i=0;i<headings.length;i++){
	const h=headings[i];if(h.level!==2 || !h.label.startsWith('Part '))continue;
	const children=[];for(let j=i+1;j<headings.length&&headings[j].level>2;j++)if(headings[j].level===3)children.push(headings[j]);
	const toc='<details class="section-toc"><summary>In this part</summary><ul>'+children.map(c=>`<li><a href="#${c.id}">${escape(c.label)}</a></li>`).join('')+'</ul></details>';
	html=html.replace(new RegExp('(<h2 id="'+h.id+'">[\\s\\S]*?<\\/h2>)'), '$1'+toc);
}
const nav=headings.filter(h=>h.level===2).map(h=>`<a href="#${h.id}">${escape(h.label)}</a>`).join('');
fs.mkdirSync(path.join(dir,'assets'),{recursive:true});
fs.cpSync(path.join(sourceAssets,'katex'),path.join(dir,'assets/katex'),{recursive:true});
fs.copyFileSync(path.join(sourceAssets,'marked-LICENSE.md'),path.join(dir,'assets/marked-LICENSE.md'));
const style=`
:root{--paper:#fcfbf8;--ink:#213431;--muted:#64736e;--accent:#286e5b;--line:#dedfd6;--soft:#edf2ec}*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:96px}body{margin:0;background:var(--paper);color:var(--ink);font:17px/1.75 Georgia,serif}a{color:var(--accent);text-underline-offset:3px}header{position:sticky;top:0;z-index:5;display:flex;align-items:center;gap:16px;padding:14px 28px;background:#fcfbf8f5;border-bottom:1px solid var(--line);font:14px/1.5 system-ui,sans-serif;flex-wrap:wrap}header strong{font-size:16px;margin-right:auto}button,.doc-link{font:inherit;background:white;color:var(--accent);border:1px solid var(--line);padding:8px 12px;border-radius:7px;cursor:pointer;text-decoration:none}button:hover,.doc-link:hover{background:var(--soft)}.layout{display:grid;grid-template-columns:270px minmax(0,960px);max-width:1320px;margin:auto;gap:42px;padding:32px 28px}nav{position:sticky;top:100px;align-self:start;max-height:calc(100vh - 120px);overflow:auto;font:13px/1.55 system-ui,sans-serif;padding-right:16px}nav a{display:block;padding:8px 10px;margin:0 0 4px;text-decoration:none;border-radius:5px}nav a:hover{background:var(--soft)}nav strong{display:block;font-size:11px;letter-spacing:.13em;color:var(--muted);margin:0 10px 15px}main{min-width:0;padding-bottom:80px}h1,h2,h3,h4{font-family:system-ui,sans-serif;line-height:1.25;letter-spacing:-.025em}h1{font-size:44px;margin:16px 0 24px}h2{font-size:30px;margin-top:64px;padding-top:12px}h3{font-size:23px;margin-top:38px}h4{font-size:19px;margin-top:25px}p{margin:16px 0}blockquote{border-left:4px solid #80a28e;margin:22px 0;background:var(--soft);padding:8px 22px;font:15px/1.75 system-ui,sans-serif}.table-wrap{overflow-x:auto;margin:22px 0}table{border-collapse:collapse;width:100%;font:14px/1.6 system-ui,sans-serif}th,td{text-align:left;vertical-align:top;padding:12px;border-bottom:1px solid var(--line);min-width:90px}th{background:var(--soft)}pre{background:#f0f1eb;border:1px solid var(--line);padding:20px;border-radius:6px;overflow:auto;font:13px/1.7 ui-monospace,monospace}code{font-size:.86em}hr{border:0;border-top:1px solid var(--line);margin:48px 0}.equation{overflow-x:auto;padding:10px 0}.section-toc{font:14px/1.6 system-ui,sans-serif;border:1px solid var(--line);border-radius:8px;padding:12px 18px;margin:20px 0}.section-toc summary{cursor:pointer;color:var(--accent)}.section-toc ul{columns:2;column-gap:30px;padding-left:20px}.section-toc li{break-inside:avoid;margin-bottom:8px}.hint{font:13px/1.6 system-ui,sans-serif;color:var(--muted)}#status{font:13px/1.5 system-ui,sans-serif;color:var(--accent)}dialog{border:1px solid var(--line);border-radius:10px;padding:24px;width:min(650px,90vw);background:var(--paper);color:var(--ink)}dialog textarea{width:100%;height:190px;font:15px/1.6 system-ui;padding:12px}dialog::backdrop{background:#102c2566}footer{font:13px/1.6 system-ui;color:var(--muted)}@media(max-width:900px){.layout{grid-template-columns:1fr;gap:15px;padding:20px}nav{position:static;max-height:220px;border-bottom:1px solid var(--line);padding-bottom:15px}h1{font-size:34px}header{padding:12px 18px;gap:8px}.section-toc ul{columns:1}}@media print{header,nav,dialog,.hint,.section-toc{display:none!important}.layout{display:block;padding:0}body{font-size:11pt;background:white}h2{break-before:page;margin-top:0}h1,h2,h3,h4{break-after:avoid}blockquote,pre{break-inside:avoid}.table-wrap,.equation{overflow:visible}a{color:inherit}}
`;
const script=`
const copyButton=document.getElementById('ask');
let lastSelection='';
document.addEventListener('selectionchange',()=>{const s=window.getSelection().toString().trim();if(s)lastSelection=s;});
copyButton.addEventListener('click',async()=>{
	const selected=lastSelection.slice(0,5000);
	const prompt=selected?'From Felipe Fritsch’s learning wiki, please explain this passage more intuitively, with a small worked example. Do not quiz me unless I ask.\\n\\n'+selected:'Please help me study mathematical foundations, beginning with vectors and projections. Use intuition and a worked example; do not require a coding task or quiz.';
	try{await navigator.clipboard.writeText(prompt);document.getElementById('status').textContent='Copied — paste into your tutor conversation.';}catch(e){document.getElementById('question').value=prompt;document.getElementById('fallback').showModal();document.getElementById('question').select();}
});
document.getElementById('close').addEventListener('click',()=>document.getElementById('fallback').close());
document.getElementById('print').addEventListener('click',()=>window.print());
`;
const output=`<!doctype html><html lang="en"><head><meta charset="utf-8"><link rel="icon" href="../favicon.svg"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Quant & AI Learning Wiki</title><link rel="stylesheet" href="assets/katex/katex.min.css"><style>${style}</style></head><body><header><strong>Felipe Fritsch · Learning Wiki</strong><a class="doc-link" href="../index.html">Home</a><a class="doc-link" href="Study_Wiki.md" download>Download notes</a><button id="ask">Copy study question</button><button id="print">Print</button><span id="status" role="status"></span></header><div class="layout"><nav aria-label="Wiki contents"><strong>READ · CONNECT · UNDERSTAND</strong>${nav}</nav><main><p class="hint">A reading-first companion. Start with the roadmap, or choose a topic from the contents.</p>${html}<footer>Felipe Fritsch · Personal study notes. No embedded chat or visitor tracking.</footer></main></div><dialog id="fallback"><h2>Copy into the tutor chat</h2><p>Automatic copying was unavailable. Copy this text manually.</p><textarea id="question" aria-label="Question for tutor" readonly></textarea><button id="close">Close</button></dialog><script>${script}</script></body></html>`;
fs.writeFileSync(path.join(dir,'index.html'),output);
const hrefs=[...output.matchAll(/href="([^"]+)"/g)].map(m=>m[1]);
const broken=hrefs.filter(x=>x.startsWith('#')&&!ids.has(x.slice(1)));
if(broken.length)throw Error('Broken internal links: '+broken.join(', '));
if(output.includes('WIKIMATHBLOCK'))throw Error('Unrendered equation');
fs.writeFileSync(path.join(dir,'render_checks.json'),JSON.stringify({equations:math.length,headings:headings.length,internalLinks:hrefs.filter(x=>x.startsWith('#')).length,brokenInternalLinks:broken,words:md.split(/\s+/).length},null,2));
console.log(`Rendered ${math.length} equations and ${headings.length} headings; internal links checked.`);
