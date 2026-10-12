import {esc, title, modelName, LANGUAGES, PRESSURES, LABELS, CATEGORIES, keyFor} from './core.js';
import {WORLD_MAP_PATH} from './world.js';

// The lab only indexes released observations. It never invents a response.
export function trialLookup(rows, manifest, questions) {
  const questionIndex=new Map(questions.map((q,i)=>[[q.scenario,q.language,q.pressure].join('|'),i]));
  const trials=new Map(rows.map(r=>[[r[0],r[1],r[2]].join('|'),r]));
  const get=c=>trials.get([manifest.models.indexOf(c.model),questionIndex.get([c.scenario,c.language,c.pressure].join('|')),manifest.prompts.indexOf(c.prompt)].join('|'));
  const config=r=>({model:manifest.models[r[0]],scenario:questions[r[1]].scenario,language:questions[r[1]].language,pressure:questions[r[1]].pressure,prompt:manifest.prompts[r[2]]});
  return {get,config};
}

export function atlasCounts(rows, manifest, questions, filters={}) {
  const blank=()=>({completed:0,errors:0,harmful:0,substantive:0,gap:0});
  const cells=Object.fromEntries(manifest.models.map(m=>[m,Object.fromEntries(manifest.languages.map(l=>[l,Object.fromEntries(manifest.prompts.map(p=>[p,blank()]))]))]));
  for(const r of rows) {
    const q=questions[r[1]];
    if(filters.category&&q.category!==filters.category||filters.pressure&&q.pressure!==filters.pressure)continue;
    const c=cells[manifest.models[r[0]]][q.language][manifest.prompts[r[2]]];
    if(!(r[4]&1)){c.errors++;continue;}
    c.completed++;
    const label=manifest.outcomes[r[3]];
    if(label==='COMPLY'||label==='SAY_DO_GAP')c.harmful++;
    if(r[4]&8)c.substantive++;
    if(label==='SAY_DO_GAP')c.gap++;
  }
  return cells;
}

export function createLab(env) {
  const {manifest:m,questions:qs,index:rows,data,navigate,routeUrl,badge,toast}=env;
  const e=esc,lookup=trialLookup(rows,m,qs),scenarioIds=[...new Set(qs.map(q=>q.scenario))].sort();
  const scenarioCategory=new Map(qs.map(q=>[q.scenario,q.category]));
  const scenarioText=new Map(qs.filter(q=>q.language==='en'&&q.pressure==='none').map(q=>[q.scenario,q.text]));
  const scenarioName=id=>/^\d+$/.test(title(id))?(scenarioText.get(id)||id).slice(0,82)+'…':title(id);
  const axes={language:'Language',prompt:'System instruction',pressure:'Social pressure',model:'Model'};
  const values={language:m.languages,prompt:m.prompts,pressure:m.pressure,model:m.models};
  const label=(axis,value)=>axis==='language'?LANGUAGES[value]:axis==='model'?modelName(value):axis==='prompt'?(value==='ethical'?'Ethical instruction':'Neutral instruction'):PRESSURES[value];
  const outcome=r=>r?m.outcomes[r[3]]:'OUTPUT_ERROR';
  const harmful=r=>['COMPLY','SAY_DO_GAP'].includes(outcome(r));
  const descriptions={NO_SUBSTANTIVE_ACTION:'No substantive tool action',COMPLY:'Harmful tool action',SAY_DO_GAP:'Objection + harmful tool action',REDIRECT:'Redirected action',AMBIGUOUS:'Ambiguous judgment',OUTPUT_ERROR:'Unscored output error'};
  const symbols={NO_SUBSTANTIVE_ACTION:'—',COMPLY:'●',SAY_DO_GAP:'↯',REDIRECT:'↗',AMBIGUOUS:'?',OUTPUT_ERROR:'!'};
  const nativeNames={en:'English',pt:'Português',es:'Español',zh:'中文',hi:'हिन्दी'};
  const anchors={en:[204,118],pt:[325,245],es:[442,104],zh:[719,113],hi:[643,153]};
  const n=v=>Number(v).toLocaleString('en-US');
  const rate=(a,b)=>b?100*a/b:null;
  const pct=v=>v===null?'N/A':v.toFixed(1)+'%';
  const signed=v=>v===null?'N/A':(v>0?'+':'')+v.toFixed(1)+' pp';
  const choice=(axis,value,id)=>`<label class="lab-field" for="${e(id)}"><span>${e(axes[axis])}</span><select id="${e(id)}" data-lab-field="${id==='lab-against'?'against':axis}">${values[axis].map(v=>`<option value="${e(v)}" ${v===value?'selected':''}>${e(label(axis,v))}</option>`).join('')}</select></label>`;
  const button=(action,text,extra='',classes='')=>`<button class="lab-button ${classes}" data-action="${action}" ${extra}>${text}</button>`;
  const route=(view,state,text,classes='lab-button')=>`<a class="${classes}" href="${e(routeUrl(view,state))}">${text}</a>`;
  const chips=(field,options,selected)=>`<div class="lab-segments" aria-label="${e(field)}">${Object.entries(options).map(([v,text])=>button('lab-set',e(text),`data-field="${e(field)}" data-value="${e(v)}" aria-pressed="${v===selected}"`)).join('')}</div>`;
  const note='<p class="lab-footnote">Recorded benchmark responses, with automated action labels. Examples illustrate individual trials; use the results for aggregate conclusions.</p>';
  let atlasCache=null,atlasKey='',quizCursor=0;

  function comparison(state) {
    const axis=Object.hasOwn(axes,state.axis)?state.axis:'prompt';
    const a={model:m.models.includes(state.model)?state.model:'llama-3.1-8b',scenario:scenarioIds.includes(state.scenario)?state.scenario:'justice_disability_exclusion',language:m.languages.includes(state.language)?state.language:'en',prompt:m.prompts.includes(state.prompt)?state.prompt:'ethical',pressure:m.pressure.includes(state.pressure)?state.pressure:'justification'};
    let against=values[axis].includes(state.against)?state.against:({language:'hi',prompt:'neutral',pressure:'none',model:'claude-haiku-4.5'})[axis];
    if(against===a[axis])against=values[axis].find(v=>v!==a[axis]);
    return {axis,a,b:{...a,[axis]:against}};
  }
  function pairState(a,b,axis){return {...a,axis,against:b[axis]};}
  async function detail(c) {
    const q=qs[lookup.get(c)?.[1]];
    if(!q)throw new Error('This condition is not in the released benchmark.');
    const shard=await data(`responses/${String(m.models.indexOf(c.model)).padStart(2,'0')}-${c.language}-${c.prompt}.json.gz`);
    const d=shard[q.id];
    if(!d)throw new Error('The recorded response is unavailable.');
    return d;
  }
  function words(text,lang) {
    const content=String(text||'');
    if(!content.trim())return '<p class="lab-empty-text">No prose was recorded.</p>';
    return `<div class="lab-prose" lang="${e(lang)}">${e(content.slice(0,650))}${content.length>650?'…':''}</div>`+(content.length>650?`<details class="lab-details"><summary>Read the full response · ${n(content.length)} characters</summary><div class="lab-prose" lang="${e(lang)}">${e(content)}</div></details>`:'');
  }
  function calls(d) {
    if(d.error||!d.mechanical.valid)return '<p class="lab-empty-text">Output error. The action outcome is not scored.</p>';
    if(!d.calls.length)return '<div class="lab-no-call"><span aria-hidden="true">—</span><p>No tool call recorded.<small>The response remains text.</small></p></div>';
    return d.calls.map((raw,i)=>{
      const call=raw.function||raw;let args=call.arguments??call.parameters??{};
      if(typeof args==='string'){try{args=JSON.parse(args);}catch{}}
      return `<details class="lab-call" ${i===0?'open':''}><summary><span class="lab-call-index">${String(i+1).padStart(2,'0')}</span><code>${e(call.name||call.tool||'Tool call')}</code><span aria-hidden="true">↗</span></summary><pre>${e(typeof args==='string'?args:JSON.stringify(args,null,2))}</pre></details>`;
    }).join('');
  }
  function request(c,d) {
    const english=qs.find(q=>q.scenario===c.scenario&&q.language==='en'&&q.pressure===c.pressure);
    return `<details class="lab-request"><summary>Read the exact request <span>${e(nativeNames[c.language])}</span></summary><p lang="${e(c.language)}">${e(d.question)}</p>${c.language!=='en'&&english?`<details><summary>English benchmark version</summary><p>${e(english.text)}</p></details>`:''}<details><summary>System instruction</summary><p>${e(d.system_prompt)}</p></details></details>`;
  }
  function responseCard(c,d,axis,side) {
    const row=lookup.get(c),o=outcome(row);
    return `<article class="lab-response lab-side-${side.toLowerCase()}" data-outcome="${o}" aria-label="Response ${side}"><div class="lab-response-head"><span class="lab-side-letter">${side}</span><div><p>${e(axes[axis])}</p><h3>${e(label(axis,c[axis]))}</h3></div>${badge(o)}</div><p class="lab-config">${e(modelName(c.model))} · ${e(LANGUAGES[c.language])} · ${e(c.prompt)} · ${e(PRESSURES[c.pressure])}</p>${request(c,d)}<section class="lab-said"><h4><span aria-hidden="true">“</span> What it said</h4>${words(d.response,c.language)}</section><section class="lab-did"><h4><span aria-hidden="true">↳</span> Proposed tool calls <span class="lab-call-count">${d.calls.length}</span></h4>${calls(d)}</section><div class="lab-response-footer"><span>${e(descriptions[o])}</span>${button('record','Inspect evidence ↗',`data-key="${e(d.key)}"`)}</div></article>`;
  }
  function world(a,b,axis) {
    const [x1,y1]=anchors[a.language],[x2,y2]=anchors[b.language];
    const arc=axis==='language'?`<path class="lab-map-route" d="M${x1},${y1} Q${(x1+x2)/2},${Math.min(y1,y2)-70} ${x2},${y2}"/>`:'';
    return `<section class="lab-map-card" aria-labelledby="map-title"><div class="lab-map-heading"><div><span class="lab-kicker">THE LANGUAGE EXPLORER</span><h3 id="map-title">One request. Five languages.</h3></div><span class="lab-map-hint">Click a language ↓</span></div><svg class="lab-world" viewBox="0 0 900 370" role="group" aria-label="Choose a prompt language. Locations are navigation anchors, not country-level results."><defs><pattern id="map-dots" width="20" height="20" patternUnits="userSpaceOnUse"><circle cx="1" cy="1" r=".7" fill="#aec9c1"/></pattern></defs><rect width="900" height="370" fill="url(#map-dots)"/><path class="lab-map-land" d="${WORLD_MAP_PATH}"/>${arc}${m.languages.map(lang=>{const [x,y]=anchors[lang],selected=lang===b.language,base=lang===a.language;return `<g id="lab-map-${lang}" class="lab-map-language ${selected?'is-b':''} ${base?'is-a':''}" transform="translate(${x},${y})" role="button" tabindex="0" data-action="lab-map" data-language="${lang}" aria-label="Compare ${e(LANGUAGES[lang])}" aria-pressed="${selected}"><title>${e(LANGUAGES[lang])} — compare recorded responses</title><rect class="map-hit" x="-56" y="-21" width="112" height="74" rx="12"/><circle class="map-halo" r="14"/><circle class="map-dot" r="5"/><rect x="-52" y="18" width="104" height="31" rx="15.5"/><text y="38" text-anchor="middle" lang="${lang}">${nativeNames[lang]}</text></g>`;}).join('')}</svg><div class="lab-map-mobile-controls" aria-label="Select a prompt language">${m.languages.map(lang=>button('lab-map',lang.toUpperCase(),`id="lab-map-mobile-${lang}" data-language="${lang}" aria-label="Select ${e(LANGUAGES[lang])}" aria-pressed="${lang===b.language}"`)).join('')}</div><div class="lab-map-footer"><span><i class="map-key-a"></i> A: ${e(LANGUAGES[a.language])}</span>${axis==='language'?`<span><i class="map-key-b"></i> B: ${e(LANGUAGES[b.language])}</span>`:''}<small>Language comparisons, not country comparisons.</small></div></section>`;
  }
  function matrix(a,b) {
    return `<section class="lab-matrix-section"><div class="lab-section-title"><div><span class="lab-kicker">KEEP EXPLORING</span><h3>Every language. Every pressure.</h3><p>${e(modelName(a.model))} · ${e(a.prompt)} instruction · this scenario. Click a cell to open its response.</p></div></div><div class="lab-matrix-scroll"><div class="lab-condition-matrix" role="group" aria-label="Observed outcomes for five languages and six pressures"><span></span>${m.pressure.map(p=>`<span class="lab-matrix-heading">${e(PRESSURES[p])}</span>`).join('')}${m.languages.map(l=>`<span class="lab-matrix-language">${e(LANGUAGES[l])}</span>`+m.pressure.map(p=>{const c={...a,language:l,pressure:p},row=lookup.get(c),o=outcome(row),selected=l===a.language&&p===a.pressure;return `<button class="lab-condition outcome-${o} ${selected?'is-selected':''}" data-action="lab-cell" data-language="${l}" data-pressure="${p}" aria-pressed="${selected}" aria-label="${e(LANGUAGES[l]+', '+PRESSURES[p]+': '+descriptions[o])}" title="${e(descriptions[o])}"><span aria-hidden="true">${symbols[o]}</span><small>${o==='SAY_DO_GAP'?'Say–do gap':o==='COMPLY'?'Harmful call':o==='NO_SUBSTANTIVE_ACTION'?'No action':LABELS[o][0]}</small></button>`;}).join('')).join('')}</div></div><div class="lab-legend">${Object.keys(descriptions).map(o=>`<span><i class="outcome-${o}">${symbols[o]}</i>${e(LABELS[o][0])}</span>`).join('')}</div><p class="lab-footnote">One recorded response per cell. “No action” means no substantive tool action; it does not establish that the response is harmless in every respect.</p></section>`;
  }
  function findContrast(axis,base,extra={}) {
    const pool=rows.filter(r=>{const c=lookup.config(r);return (axis==='model'||c.model===base.model)&&(!extra.language||c.language===extra.language)&&(!extra.category||qs[r[1]].category===extra.category)&&(!extra.pressure||c.pressure===extra.pressure)&&(r[4]&1);});
    let fallback=null;
    for(let i=0;i<pool.length;i++){
      const row=pool[(i+quizCursor*137)%pool.length],a=lookup.config(row);
      for(const v of values[axis]){
        if(v===a[axis])continue;
        const b={...a,[axis]:v},other=lookup.get(b);
        if(!other||!(other[4]&1)||harmful(row)===harmful(other))continue;
        const result=pairState(a,b,axis);
        if(outcome(row)==='SAY_DO_GAP'||outcome(other)==='SAY_DO_GAP')return result;
        fallback??=result;
      }
    }
    return fallback;
  }
  async function play(state) {
    const {axis,a,b}=comparison(state),[da,db]=await Promise.all([detail(a),detail(b)]),ra=lookup.get(a),rb=lookup.get(b);
    const baseText=qs.find(q=>q.scenario===a.scenario&&q.language==='en'&&q.pressure==='none')?.text||'';
    const different=harmful(ra)!==harmful(rb),gap=[outcome(ra),outcome(rb)].includes('SAY_DO_GAP');
    const outcomes=[outcome(ra),outcome(rb)];
    const story=outcomes.includes('OUTPUT_ERROR')?'An output error leaves one action outcome unscored.':outcomes.includes('AMBIGUOUS')?'At least one of these action judgments is ambiguous.':gap?'A refusal in words can still accompany a harmful call.':different?'One changed setting. A different action outcome.':'These two trials have the same harmful-action status.';
    return `<section class="lab-intro"><div><p class="lab-kicker">EXPLORE THE BENCHMARK</p><h2>Change one thing.<br><em>Look at what the model does.</em></h2><p>Choose a language on the map, change an instruction, or compare models.<br class="desktop-break"> Read the exact words and the tool calls side by side.</p></div><div class="lab-trial-stamp"><strong>194,400</strong><span>recorded trials to explore</span><small>30 models · 108 scenarios · 5 languages</small></div></section><div class="lab-story-buttons" aria-label="Guided examples">${button('lab-story','<span>01</span> The say–do gap','data-story="gap"')}${button('lab-story','<span>02</span> Across languages','data-story="language"')}${button('lab-story','<span>03</span> Under pressure','data-story="pressure"')}${route('guess',{},'<span>04</span> Predict the action')}</div><div class="lab-stage">${world(a,b,axis)}<aside class="lab-controls"><span class="lab-kicker">YOUR COMPARISON</span><label class="lab-field" for="lab-scenario"><span>Choose a scenario</span><select id="lab-scenario" data-lab-field="scenario">${Object.entries(CATEGORIES).map(([cat,name])=>`<optgroup label="${e(name)}">${scenarioIds.filter(s=>scenarioCategory.get(s)===cat).map(s=>`<option value="${s}" ${s===a.scenario?'selected':''}>${e(scenarioName(s))}</option>`).join('')}</optgroup>`).join('')}</select></label><p class="lab-scenario-text">${e(baseText)}</p><span class="lab-control-label">Change only this factor</span>${chips('axis',axes,axis)}<div class="lab-axis-choices">${choice(axis,a[axis],'lab-'+axis)}<span aria-hidden="true">→</span>${choice(axis,b[axis],'lab-against')}</div><details class="lab-fixed"><summary>Shared settings <span>3 held constant</span></summary><div class="lab-shared-controls">${Object.keys(axes).filter(k=>k!==axis).map(k=>choice(k,a[k],'lab-'+k)).join('')}</div></details><div class="lab-control-actions">${button('lab-contrast','Find another contrast ↗')}${button('lab-share','Copy link','', 'lab-quiet')}</div></aside></div><div class="lab-comparison-summary" role="status"><span class="lab-summary-icon" aria-hidden="true">${gap?'↯':different?'⇄':'='}</span><div><h3>${story}</h3><p>Only <strong>${e(axes[axis].toLowerCase())}</strong> differs: ${e(label(axis,a[axis]))} → ${e(label(axis,b[axis]))}. The scenario and other displayed settings are matched.</p></div></div><div class="lab-response-pair">${responseCard(a,da,axis,'A')}${responseCard(b,db,axis,'B')}</div>${note}${matrix(a,b)}<div class="lab-next"><h3>Is this a single example or a wider pattern?</h3>${route('atlas',{model:a.model,language:a.language},'Explore the model × language atlas →')}</div>`;
  }
  function metric(c,type){return type==='gap'?rate(c.gap,c.substantive):rate(c.harmful,c.completed);}
  function atlas(state) {
    const prompt=['ethical','neutral','difference'].includes(state.prompt)?state.prompt:'ethical',type=state.metric==='gap'?'gap':'harmful';
    const filters={category:m.categories.includes(state.category)?state.category:'',pressure:m.pressure.includes(state.pressure)?state.pressure:''};
    const key=JSON.stringify(filters);
    if(key!==atlasKey||!atlasCache){atlasCache=atlasCounts(rows,m,qs,filters);atlasKey=key;}
    const counts=atlasCache,selectedModel=m.models.includes(state.model)?state.model:'llama-3.1-8b',selectedLang=m.languages.includes(state.language)?state.language:'en';
    const value=(model,lang)=>{const c=counts[model][lang];if(prompt!=='difference')return metric(c[prompt],type);const a=metric(c.ethical,type),b=metric(c.neutral,type);return a===null||b===null?null:b-a;};
    const models=[...m.models];if(state.sort!=='name')models.sort((a,b)=>{const average=x=>{const rates=m.languages.map(l=>value(x,l)).filter(v=>v!==null);return rates.length?rates.reduce((s,v)=>s+v,0)/rates.length:-Infinity;};return average(b)-average(a);});
    const cellTitle=(model,lang)=>{const c=counts[model][lang];return `${modelName(model)} · ${LANGUAGES[lang]}: `+m.prompts.map(p=>`${p}: ${type==='gap'?c[p].gap:c[p].harmful}/${type==='gap'?c[p].substantive:c[p].completed} (${pct(metric(c[p],type))}), ${c[p].errors} output errors`).join('; ');};
    const selected=counts[selectedModel][selectedLang],left=metric(selected.ethical,type),right=metric(selected.neutral,type);
    return `<section class="lab-intro lab-intro-compact"><div><p class="lab-kicker">THE INTERACTIVE ATLAS</p><h2>Where do the <em>gaps appear?</em></h2><p>Compare every model across languages. Click a cell to inspect its counts and open the responses behind it.</p></div>${route('play',{},'← Back to the playground')}</section><div class="atlas-controls"><div><span class="lab-control-label">Measure</span>${chips('metric',{harmful:'Harmful tool calls',gap:'Say–do gap'},type)}</div><div><span class="lab-control-label">System instruction</span>${chips('prompt',{ethical:'Ethical',neutral:'Neutral',difference:'Neutral − ethical'},prompt)}</div><label class="lab-field" for="atlas-category"><span>Scenario category</span><select id="atlas-category" data-lab-field="category"><option value="">All 108 scenarios</option>${m.categories.map(c=>`<option value="${c}" ${filters.category===c?'selected':''}>${e(CATEGORIES[c])}</option>`).join('')}</select></label><label class="lab-field" for="atlas-pressure"><span>Social pressure</span><select id="atlas-pressure" data-lab-field="pressure"><option value="">All six conditions</option>${m.pressure.map(p=>`<option value="${p}" ${filters.pressure===p?'selected':''}>${e(PRESSURES[p])}</option>`).join('')}</select></label></div><div class="atlas-layout"><section class="atlas-board"><div class="atlas-board-title"><span>${type==='gap'?'Objection with a harmful call / substantive-call responses':'Responses with harmful calls / completed responses'}</span>${button('lab-sort',state.sort==='name'?'Sort by rate ↓':'Sort by name A–Z')}</div><div class="atlas-grid" role="group" aria-label="Model by language rates"><span class="atlas-column-label">MODEL</span>${m.languages.map(l=>`<span class="atlas-column-label"><b>${l.toUpperCase()}</b><small>${e(LANGUAGES[l])}</small></span>`).join('')}${models.map(model=>`<span class="atlas-model-label ${model===selectedModel?'is-selected':''}">${e(modelName(model))}</span>`+m.languages.map(lang=>{const v=value(model,lang),level=v===null?0:Math.min(9,Math.ceil(Math.abs(v)/10)),selected=model===selectedModel&&lang===selectedLang;return `<button class="atlas-cell heat-${type} heat-${level} ${v!==null&&v<0?'heat-negative':''} ${v===null?'is-unavailable':''} ${selected?'is-selected':''}" data-action="lab-atlas-cell" data-model="${e(model)}" data-language="${lang}" aria-pressed="${selected}" aria-label="${e(cellTitle(model,lang))}" title="${e(cellTitle(model,lang))}">${v===null?'—':(prompt==='difference'&&v>0?'+':'')+v.toFixed(1)}<small>${prompt==='difference'?'pp':'%'}</small></button>`;}).join('')).join('')}</div><div class="atlas-legend"><span>Lower rate</span>${Array.from({length:10},(_,i)=>`<i class="heat-${type} heat-${i}"></i>`).join('')}<span>Higher rate</span></div><p class="lab-footnote">${prompt==='difference'?'Differences are percentage points, not relative percentage changes. Blue cells indicate lower rates under neutral instructions. ':''}Output errors are excluded from rate denominators. A dash means no eligible denominator. These filtered summaries are descriptive.</p></section><aside class="atlas-detail" aria-live="polite"><span class="lab-kicker">SELECTED CELL</span><h3>${e(modelName(selectedModel))}</h3><span class="atlas-language-name">${e(LANGUAGES[selectedLang])}</span><div class="atlas-rate-pair">${m.prompts.map(p=>`<div><span>${p==='ethical'?'Ethical':'Neutral'}</span><strong>${pct(metric(selected[p],type))}</strong><small>${n(type==='gap'?selected[p].gap:selected[p].harmful)} / ${n(type==='gap'?selected[p].substantive:selected[p].completed)}</small></div>`).join('')}</div><div class="atlas-difference"><strong>${signed(left===null||right===null?null:right-left)}</strong><span>neutral − ethical</span></div><p>${type==='gap'?'A say–do gap is an objection or refusal paired with a harmful tool call. Its denominator includes only substantive-call responses.':'Harmful calls include compliance and say–do gaps. A refusal in prose does not cancel a harmful call.'}</p><p class="lab-footnote">${n(selected.ethical.errors+selected.neutral.errors)} output errors in this selection, across both instructions.</p>${button('lab-atlas-compare','Open a matched comparison →',`data-model="${e(selectedModel)}" data-language="${selectedLang}"`)}${route('responses',{model:selectedModel,language:selectedLang,prompt:prompt==='difference'?'':prompt,...filters},'Browse all underlying responses ↗','lab-text-link')}<div class="atlas-guidance"><strong>What changes when you click?</strong><p>The panel shows the exact numerators and denominators. Open a comparison to read two matched responses.</p></div></aside></div>`;
  }
  const quizPools={gap:rows.filter(r=>outcome(r)==='SAY_DO_GAP'&&qs[r[1]].language==='en'),comply:rows.filter(r=>outcome(r)==='COMPLY'&&qs[r[1]].language==='en'),none:rows.filter(r=>outcome(r)==='NO_SUBSTANTIVE_ACTION'&&qs[r[1]].language==='en')};
  function quizRow(state){const byKey=state.case?rows.find(r=>keyFor(r,m,qs)===state.case):null;if(byKey&&(byKey[4]&1)&&['NO_SUBSTANTIVE_ACTION','COMPLY','SAY_DO_GAP'].includes(outcome(byKey)))return byKey;return lookup.get({model:'llama-3.1-8b',scenario:'justice_disability_exclusion',language:'en',prompt:'neutral',pressure:'justification'});}
  async function guess(state) {
    const row=quizRow(state),c=lookup.config(row),d=await detail(c),o=outcome(row),revealed=['harmful','not-harmful'].includes(state.guess),correct=(state.guess==='harmful')===harmful(row);
    return `<section class="lab-intro lab-intro-compact"><div><p class="lab-kicker">READ BETWEEN WORDS AND ACTIONS</p><h2>Would you <em>spot the gap?</em></h2><p>Read the model’s words. Then predict whether this recorded response also contains a harmful tool call.</p></div>${route('play',{},'← Back to the playground')}</section><div class="guess-layout"><article class="guess-words"><span class="lab-kicker">THE REQUEST</span><h3>${e(scenarioName(c.scenario))}</h3><p class="guess-request" lang="${c.language}">${e(d.question)}</p><div class="guess-model"><span class="lab-avatar" aria-hidden="true">L</span><div><strong>${e(modelName(c.model))}</strong><span>${e(LANGUAGES[c.language])} · ${e(c.prompt)} instruction · ${e(PRESSURES[c.pressure])}</span></div></div>${words(d.response,c.language)}<p class="lab-footnote">The prose is shown exactly as recorded. Tool calls are revealed after your prediction.</p></article><aside class="guess-panel">${revealed?`<span class="lab-kicker">THE RECORDED OUTCOME</span><h3>${correct?'You spotted it.':'The action tells another story.'}</h3>${badge(o)}<p>${e(descriptions[o])}. ${o==='SAY_DO_GAP'?'The objection in the text coexists with a tool call judged to carry out the harmful request.':o==='NO_SUBSTANTIVE_ACTION'?'This record has no substantive tool action.':'This is the stored benchmark outcome, not a new judgment.'}</p><section class="lab-did"><h4>Proposed tool calls</h4>${calls(d)}</section><div class="guess-actions">${button('lab-next-case','Try another recorded response →')}${button('record','Inspect the complete evidence ↗',`data-key="${e(d.key)}"`)}${route('play',{...c,axis:'prompt',against:c.prompt==='ethical'?'neutral':'ethical'},'Compare the other instruction ↗','lab-text-link')}</div>`:`<span class="guess-question-mark" aria-hidden="true">?</span><span class="lab-kicker">YOUR PREDICTION</span><h3>What happened<br>after these words?</h3><p>Would you expect a harmful tool call in the same response?</p><div class="guess-actions">${button('lab-guess','No harmful tool call','data-guess="not-harmful"')}${button('lab-guess','A harmful tool call','data-guess="harmful"')}</div><span class="guess-reveal-note">Choose to reveal the actual tool calls.</span>`}<p class="lab-footnote">Selected examples for exploration, not a representative sample or a test of your safety expertise.</p></aside></div>`;
  }
  function set(field,value,state,view) {
    const next={...state,[field]:value,record:null};
    if(field==='axis')delete next.against;
    if(field==='model'&&view==='play')delete next.against;
    navigate(next,view,true);
  }
  function action(action,el,state,view) {
    if(!action.startsWith('lab-'))return false;
    const {axis,a,b}=comparison(state);
    if(action==='lab-set')set(el.dataset.field,el.dataset.value,state,view);
    if(action==='lab-map'){
      const lang=el.dataset.language,first=a.language===lang?(b.language!==lang?b.language:m.languages.find(l=>l!==lang)):a.language;
      navigate({...a,language:first,axis:'language',against:lang},'play',true);
    }
    if(action==='lab-cell')navigate({...pairState(a,b,axis),language:el.dataset.language,pressure:el.dataset.pressure},'play',true);
    if(action==='lab-story'){
      if(el.dataset.story==='gap')navigate({},'play',true);
      else{quizCursor++;const target=findContrast(el.dataset.story,{...a,model:'llama-3.1-8b'});if(target)navigate(target,'play',true);}
    }
    if(action==='lab-contrast'){
      quizCursor++;const target=findContrast(axis,a);
      if(target)navigate(target,'play',true);else toast('No harmful-action contrast was found for this model and factor. Try another model.');
    }
    if(action==='lab-share'){
      navigate(pairState(a,b,axis),'play',true);
      try{navigator.clipboard.writeText(location.origin+location.pathname+routeUrl('play',pairState(a,b,axis))).then(()=>toast('Comparison link copied.'),()=>toast('Copy the comparison link from your address bar.'));}catch{toast('Copy the comparison link from your address bar.');}
    }
    if(action==='lab-sort')set('sort',state.sort==='name'?'rate':'name',state,view);
    if(action==='lab-atlas-cell')navigate({...state,model:el.dataset.model,language:el.dataset.language},'atlas',true);
    if(action==='lab-atlas-compare'){
      const base={...a,model:el.dataset.model,language:el.dataset.language};
      const target=findContrast('prompt',base,{language:base.language,category:state.category,pressure:state.pressure});
      if(target)navigate(target,'play');else{
        const q=qs.find(q=>q.language===base.language&&(!state.category||q.category===state.category)&&(!state.pressure||q.pressure===state.pressure));
        navigate({...base,scenario:q.scenario,pressure:q.pressure,prompt:'ethical',axis:'prompt',against:'neutral'},'play');
      }
    }
    if(action==='lab-guess')navigate({case:keyFor(quizRow(state),m,qs),guess:el.dataset.guess},'guess',true);
    if(action==='lab-next-case'){
      const pool=quizPools[['none','gap','comply'][quizCursor++%3]],row=pool[Math.floor(Math.random()*pool.length)];
      navigate({case:keyFor(row,m,qs)},'guess',true);
    }
    return true;
  }
  return {render:(view,state)=>({play,atlas,guess})[view](state),set,action};
}
