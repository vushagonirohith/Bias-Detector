const $ = (id) => document.getElementById(id);

const state = {
  examples: {},
  history: []
};

function escapeHtml(s){ return s.replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }

function renderTextWithHighlights(text, spans){
  spans = (spans||[]).slice().sort((a,b)=>a.start-b.start);
  const merged = [];
  for(const s of spans){
    if (!merged.length || s.start >= merged[merged.length-1].end){
      merged.push({...s});
    } else {
      const last = merged[merged.length-1];
      last.end = Math.max(last.end, s.end);
      last.score = Math.max(last.score, s.score);
    }
  }
  let out = '', i = 0;
  const maxScore = merged[0]?.score || 1;
  for(const s of merged){
    if (i < s.start) out += escapeHtml(text.slice(i, s.start));
    const seg = text.slice(s.start, s.end);
    const alpha = Math.min(1, 0.25 + (s.score / maxScore) * 0.75);
    out += `<mark style="background:rgba(255,107,107,${alpha.toFixed(2)})" title="${s.score.toFixed(3)}">${escapeHtml(seg)}</mark>`;
    i = s.end;
  }
  if (i < text.length) out += escapeHtml(text.slice(i));
  return out;
}

function renderExamples(examples){
  const wrap = $('examples');
  wrap.innerHTML = '';
  Object.entries(examples).forEach(([group, items])=>{
    const h = document.createElement('div');
    h.className = 'row';
    h.innerHTML = `<div class="tag">${escapeHtml(group)}</div>`;
    wrap.appendChild(h);
    const row = document.createElement('div');
    row.className = 'chips';
    items.forEach(txt=>{
      const chip = document.createElement('div');
      chip.className = 'chip';
      chip.textContent = txt;
      chip.onclick = ()=> { $('input').value = txt; analyze(); };
      row.appendChild(chip);
    });
    wrap.appendChild(row);
  });
}

function renderProbs(probs){
  const container = $('probs');
  container.innerHTML = '';
  const entries = Object.entries(probs).sort((a,b)=>b[1]-a[1]);
  const top = entries[0]?.[1] || 1;
  entries.forEach(([k,v])=>{
    const line = document.createElement('div');
    line.className = 'prob';
    const label = document.createElement('div');
    label.className = 'label';
    label.textContent = k;
    const meter = document.createElement('div');
    meter.className = 'meter';
    const fill = document.createElement('span');
    fill.style.width = `${(v*100).toFixed(0)}%`;
    meter.appendChild(fill);
    const val = document.createElement('div');
    val.className = 'val';
    val.textContent = `${(v*100).toFixed(2)}%`;
    line.appendChild(label); line.appendChild(meter); line.appendChild(val);
    container.appendChild(line);
  });
}

function pushHistory(entry){
  state.history.unshift(entry);
  state.history = state.history.slice(0, 8);
  const h = $('history');
  h.innerHTML = '';
  state.history.forEach(item=>{
    const div = document.createElement('div');
    div.className = 'item';
    div.innerHTML = `<div class="tag">${escapeHtml(item.bias_type)}</div><div class="small">${escapeHtml(item.text.slice(0, 60))}${item.text.length>60?'…':''}</div>`;
    div.onclick = () => {
      $('input').value = item.text;
      renderResult(item.data);
    };
    h.appendChild(div);
  });
}

function toast(msg){
  const t = $('toast');
  t.textContent = msg;
  t.classList.remove('hidden');
  setTimeout(()=> t.classList.add('hidden'), 2000);
}

function copyText(text){
  navigator.clipboard.writeText(text).then(()=>toast('Copied!'));
}

function renderResult(data){
  $('result').classList.remove('hidden');
  $('bias').textContent = data.bias_type;
  $('conf').style.width = `${(data.confidence*100).toFixed(0)}%`;
  $('original').innerHTML = renderTextWithHighlights(data.original_text, data.highlights || []);
  $('neutral').textContent = data.neutralized_text || '';
  renderProbs(data.probs || {});
}

async function analyze(){
  const text = $('input').value.trim();
  if (!text) return toast('Enter some text first');
  $('loading').classList.remove('hidden');
  $('result').classList.add('hidden');

  const threshold = parseFloat($('threshold').value);
  const neutralize = $('neutralize').checked;

  try{
    const res = await fetch('/api/predict', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ text, threshold, neutralize })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data?.error || 'Request failed');
    renderResult(data);
    pushHistory({ text, data, bias_type: data.bias_type });
  }catch(e){
    toast(e.message);
  }finally{
    $('loading').classList.add('hidden');
  }
}

function init(){
  // controls
  $('run').onclick = analyze;
  $('clear').onclick = ()=>{ $('input').value=''; };
  $('copyOriginal').onclick = ()=> copyText($('input').value);
  $('copyNeutral').onclick = ()=> copyText($('neutral').textContent);
  $('threshold').oninput = (e)=> { $('thresholdOut').textContent = parseFloat(e.target.value).toFixed(2); };

  // load examples
  fetch('/api/examples').then(r=>r.json()).then(data=>{
    state.examples = data; renderExamples(data);
  });

  // show initial threshold
  $('thresholdOut').textContent = parseFloat($('threshold').value).toFixed(2);
}

document.addEventListener('DOMContentLoaded', init);
