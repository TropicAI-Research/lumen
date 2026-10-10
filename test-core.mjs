// Run with `node site/test-core.mjs` from the repository root.
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {gunzipSync} from 'node:zlib';
import {esc,matchRecord,sortRecords,keyFor,csv,summarizeResults,benignResultCounts} from './core.js';
const json=path=>JSON.parse(readFileSync(new URL(path,import.meta.url)));
const gz=path=>JSON.parse(gunzipSync(readFileSync(new URL(path,import.meta.url))));
const m=json('data/manifest.json'),q=gz('data/questions.json.gz'),rows=gz('data/index.json.gz');
const matches=state=>rows.filter(r=>matchRecord(r,state,m,q));
assert.equal(matches({}).length,194400);
assert.equal(matches({model:'claude-haiku-4.5',language:'en',prompt:'ethical'}).length,648);
assert.equal(matches({label:'OUTPUT_ERROR'}).length,32);
assert.equal(matches({label:'SAY_DO_GAP'}).length,1441);
assert.equal(matches({label:'HARMFUL'}).length,15940);
assert.equal(matches({evidence:'cross'}).length,597);
assert.equal(matches({evidence:'human'}).length,83);
assert.equal(matches({pressure:'none'}).length,32400);
assert.equal(matches({scenario:'justice_exclude_poor_candidates'}).length,1800);
assert.equal(matches({q:'not-a-real-request-123123'}).length,0);
const row=sortRecords(rows,'evidence',m,q)[0];
assert.equal(m.outcomes[row[3]],'SAY_DO_GAP');
assert.equal(keyFor(row,m,q).split('|').length,4);
assert.equal(esc('<script>"&\''),'&lt;script&gt;&quot;&amp;&#39;');
assert.equal(csv([['=SUM(A1)', 'a"b', '<img>']]),'"\'=SUM(A1)","a""b","<img>"');
console.log('Explorer index, filters, sorting, HTML escaping, and CSV safety: passed.');

const summary=summarizeResults(rows,m,q),stats=gz('data/statistics.json.gz'),benign=gz('data/benign.json.gz');
assert.equal(summary.total.trials,m.total);
assert.equal(summary.total.completed,stats.audit.n_cells);
assert.equal(summary.total.harmful,stats.audit.counts.harmful);
assert.equal(summary.total.sdg,stats.audit.counts.sdg);
assert.equal(summary.total.substantive,stats.audit.counts.l2);
assert.equal(summary.total.errors,32);
assert.equal((100*summary.prompts.ethical.harmful/summary.prompts.ethical.completed).toFixed(2),'2.76');
assert.equal((100*summary.prompts.neutral.harmful/summary.prompts.neutral.completed).toFixed(2),'13.64');
for(const model of m.models) {
  const expected=m.model_counts[model],actual=summary.models[model];
  assert.equal(actual.total.harmful,(expected.COMPLY||0)+(expected.SAY_DO_GAP||0));
  assert.equal(actual.total.trials,6480);
  assert.equal(actual.total.errors,expected.OUTPUT_ERROR||0);
  for(const field of Object.keys(summary.total)) assert.equal(actual.prompts.ethical[field]+actual.prompts.neutral[field],actual.total[field]);
}
assert.equal(summary.models['gemini-3.1-pro-preview'].total.substantive,0);
// Even a malformed row marked with a harmful label must not count as an action.
const invalid=[...rows[0]];invalid[3]=m.outcomes.indexOf('COMPLY');invalid[4]=0;
assert.equal(summarizeResults([invalid],m,q).total.harmful,0);
for(const row of benign.statistics.by_model) {
  const actual=benignResultCounts(row),source=benign.rows.filter(r=>r.model===row.model);
  assert.equal(actual.trials,source.length);
  assert.equal(actual.completed,source.filter(r=>r.benign_success).length);
  assert.equal(actual.harmful,source.filter(r=>r.harmful_violation).length);
  assert.equal(actual.conditionalHarmful,source.filter(r=>r.benign_success&&r.harmful_violation).length);
}
for(const row of benign.statistics.by_language_prompt) {
  const actual=benignResultCounts(row),source=benign.rows.filter(r=>r.language===row.language&&r.prompt_type===row.prompt_type);
  assert.equal(actual.trials,source.length);
  assert.equal(actual.completed,source.filter(r=>r.benign_success).length);
  assert.equal(actual.conditionalHarmful,source.filter(r=>r.benign_success&&r.harmful_violation).length);
}
const b=benignResultCounts(benign.statistics.overall_paired);
assert.deepEqual(b,{trials:3300,completed:2733,available:3300,capable:2733,harmful:313,conditionalHarmful:271});
assert.equal((100*b.conditionalHarmful/b.capable).toFixed(2),'9.92');
assert.equal(stats.comparisons.harmful.length,8);
assert.deepEqual(stats.comparisons.harmful.filter(r=>r.p_holm<.05).map(r=>r.prompt+':'+r.language),['ethical:hi','ethical:pt','neutral:hi','neutral:es']);
console.log('Results: every model, both prompts, benign pair counts, ten language/prompt cells, and all eight paired tests match the released data.');
