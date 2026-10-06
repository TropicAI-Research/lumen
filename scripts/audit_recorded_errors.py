#!/usr/bin/env python3
"""Inventory legacy client-error sentinels without changing source records."""
from collections import Counter
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from metrics_reconciled import is_api_error
from scripts.canonical_panel import DEFECT_IDS, digest, key_for


def main():
    rows = []
    questions = {q['question_id']:q for lang in ['pt','en','es','zh','hi']
                 for q in json.loads((ROOT/f'questions_v3_{lang}.json').read_text())}
    for path in sorted((ROOT/'eval_results_consolidated').glob('*/*/*/consolidated.json')):
        lang,prompt,model = path.parts[-4:-1]
        for record in json.loads(path.read_text()):
            if not is_api_error(record):
                continue
            text = record.get('assistant_response') or ''
            match = re.search(r'Error code:\s*(\d{3})', text)
            q = questions[record['question_id']]
            rows.append({'key':key_for(model,lang,prompt,record['question_id']),
                'model':model,'language':lang,'prompt':prompt,'scenario_id':q['scenario_id'],
                'source_path':str(path.relative_to(ROOT)),'source_record_sha256':digest(record),
                'explicit_error_field':bool(record.get('error')),
                'client_error_sentinel':text.startswith('[ERROR]'),
                'http_status_in_text':int(match.group(1)) if match else None,
                'in_common_scenarios':q['scenario_id'] not in DEFECT_IDS,
                'partial_tool_history':bool(record.get('tools_called'))})
    report={'original_records_unchanged':True,'n_error_records':len(rows),
            'n_in_common_scenarios':sum(r['in_common_scenarios'] for r in rows),
            'by_model':dict(Counter(r['model'] for r in rows)),
            'interpretation':'Client failures are unknown outcomes, including sessions with partial tool history; no refusal or completion is inferred.',
            'records':rows}
    path=ROOT/'analysis/verification/legacy_api_error_audit.json'
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='records'}))


if __name__=='__main__':
    main()
