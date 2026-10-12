#!/usr/bin/env python3
"""Local read-only browser smoke checks; requires Playwright and a served site."""
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright, expect


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',default='http://127.0.0.1:8768/')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    errors,checks=[],[]
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,executable_path='/usr/bin/google-chrome',args=['--no-sandbox'])
        context=browser.new_context(viewport={'width':1440,'height':1000},accept_downloads=True)
        page=context.new_page()
        page.on('pageerror',lambda error:errors.append(str(error)))
        page.on('console',lambda message:errors.append(message.text) if message.type=='error' else None)
        requests=[]
        page.on('request',lambda request:requests.append(request.url))

        def go(fragment):
            page.goto(args.url+'#'+fragment,wait_until='networkidle')
            expect(page.locator('#app')).to_have_attribute('aria-busy','false',timeout=60000)

        go('results')
        expect(page.locator('[data-view=results]')).to_have_attribute('aria-current','page')
        expect(page.locator('.results-headlines')).to_contain_text('2.76%')
        expect(page.locator('.results-headlines')).to_contain_text('13.64%')
        expect(page.locator('.results-headlines')).to_contain_text('8.57%')
        expect(page.locator('.results-headlines')).to_contain_text('82.82%')
        expect(page.locator('.results-body .table-wrap').first.locator('tbody tr')).to_have_count(31)
        page.screenshot(path=str(args.output/'results-desktop.png'),full_page=False)
        checks.append('Results view displays paper results and all 30 models')
        go('results?section=benign')
        expect(page.locator('.results-body')).to_contain_text('9.92%')
        expect(page.locator('.results-body')).to_contain_text('4.62%–16.32%')
        expect(page.locator('.results-body .table-wrap').nth(1).locator('tbody tr')).to_have_count(31)
        expect(page.locator('.results-body .table-wrap').nth(1).locator('tr',has_text='Gemini 3.1 Pro')).to_contain_text('110 / 110')
        expect(page.locator('.results-body .table-wrap').nth(1).locator('tr',has_text='Claude Haiku 4.5')).to_contain_text('61 / 110')
        page.screenshot(path=str(args.output/'benign-results-desktop.png'),full_page=True)
        with page.expect_download() as download:
            page.locator('[data-action=download]').click()
        target=args.output/'benign-results.csv'
        download.value.save_as(target)
        assert len(target.read_text().splitlines())==31
        checks.append('Benign results show 30 models, paired outcomes, confidence interval, and CSV export')
        page.locator('.results-pairs a',has_text='2,462 pairs').click()
        expect(page.locator('.result-bar')).to_contain_text('2,462 records')
        checks.append('A result count opens its exact underlying benign trials')
        go('results?section=languages')
        expect(page.locator('.results-body tbody tr')).to_have_count(8)
        expect(page.locator('.results-body')).to_contain_text('0.0059')
        assert 'NaN' not in page.locator('#app').inner_text()
        checks.append('Eight within-model language comparisons load from the published tests')
        go('responses')
        expect(page.locator('tbody tr')).to_have_count(20)
        expect(page.locator('.result-bar')).to_contain_text('194,400')
        page.screenshot(path=str(args.output/'desktop.png'),full_page=False)
        checks.append('Full 194,400-trial index loads; 20-row pagination')
        page.locator('#filter-label').select_option('OUTPUT_ERROR')
        expect(page.locator('.result-bar')).to_contain_text('32 records')
        page.locator('tbody .row-title').first.click()
        expect(page.locator('#record-content')).to_contain_text('Output error — not a refusal.',timeout=15000)
        expect(page.locator('#record-content')).not_to_contain_text('Record unavailable')
        page.keyboard.press('Escape')
        expect(page.locator('dialog')).not_to_be_visible()
        checks.append('Output-error filter and detail keep errors separate from refusals')
        page.locator('[data-action=reset]').click()
        page.locator('#filter-model').select_option('claude-haiku-4.5')
        page.locator('#filter-language').select_option('en')
        page.locator('#filter-prompt').select_option('ethical')
        expect(page.locator('.result-bar')).to_contain_text('648 records')
        with page.expect_download() as download:
            page.locator('[data-action=download]').click()
        target=args.output/'filtered.csv'
        download.value.save_as(target)
        assert len(target.read_text().splitlines())==649
        checks.append('Combined filters and downloadable 648-record CSV')
        page.locator('#filter-q').fill('no-match-9385829358')
        expect(page.locator('.empty')).to_contain_text('No matching records')
        page.locator('.empty [data-action=reset]').click()
        expect(page.locator('tbody tr')).to_have_count(20)
        page.locator('tbody .row-title').first.click()
        expect(page.locator('#record-title')).to_contain_text('Gemini',timeout=15000)
        expect(page.locator('#record-content')).to_contain_text('Semantic judgment')
        expect(page.locator('#record-content')).to_contain_text('Sandbox replay')
        expect(page.locator('#record-content')).to_contain_text('Full source record SHA-256')
        share=page.url
        page.reload(wait_until='networkidle')
        expect(page.locator('#record-title')).to_contain_text('Gemini',timeout=15000)
        assert page.url==share
        page.screenshot(path=str(args.output/'response-detail.png'))
        checks.append('Record deep link survives reload; calls, judge and sandbox visible')
        go('scenarios')
        expect(page.locator('.cards .card')).to_have_count(20)
        page.locator('.cards [data-action=record]').first.click()
        expect(page.locator('.translation-grid .record-panel')).to_have_count(5,timeout=15000)
        page.locator('dialog #filter-pressure').select_option('urgency')
        expect(page.locator('.translation-grid .record-panel')).to_have_count(5)
        expect(page.locator('.source-language')).to_contain_text('Portuguese')
        checks.append('Scenario comparison shows all five languages and switches pressure')
        go('judges?agreement=disagree')
        expect(page.locator('.result-bar')).to_contain_text('54 records')
        page.locator('tbody .row-title').first.click()
        expect(page.locator('.judge-cards>div')).to_have_count(3,timeout=15000)
        expect(page.locator('#record-content')).to_contain_text('Labels disagree')
        page.screenshot(path=str(args.output/'judge-comparison.png'))
        checks.append('Cross-judge disagreements show all three actual labels')
        go('judges?sample=human')
        expect(page.locator('.result-bar')).to_contain_text('83 records')
        page.locator('tbody .row-title').first.click()
        expect(page.locator('.judge-cards>div')).to_have_count(2,timeout=15000)
        expect(page.locator('#record-content')).to_contain_text('Human author')
        checks.append('83-case Portuguese human calibration is separate from model judges')
        go('benign?model=gemini-3.1-pro-preview')
        expect(page.locator('.result-bar')).to_contain_text('110 records')
        expect(page.locator('.mini-stats')).to_contain_text('100.00%')
        page.locator('tbody .row-title').first.click()
        expect(page.locator('#record-content')).to_contain_text('Mechanical completion criteria',timeout=15000)
        page.locator('dialog a',has_text='Open harmful counterpart').click()
        expect(page.locator('#record-content')).to_contain_text('Semantic judgment',timeout=15000)
        expect(page.locator('#record-content')).to_contain_text('Gemini 3.1 Pro')
        checks.append('Benign criteria and exact harmful counterpart are navigable')
        go('about')
        expect(page.locator('.tool-list details')).to_have_count(21)
        assert 'NaN' not in page.locator('#app').inner_text()
        page.locator('.tool-list summary').first.click()
        expect(page.locator('.tool-list details[open] pre')).to_be_visible()
        checks.append('21 schemas, paired tests and confusion matrix load without invalid values')
        assert all(url.startswith(args.url.split('#')[0].rstrip('/')) for url in requests),requests
        checks.append('No third-party runtime requests or paid API calls')
        for width in [390,320,768]:
            page.set_viewport_size({'width':width,'height':844})
            for fragment in ['results','results?section=benign','results?section=languages']:
                go(fragment)
                assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'),(width,fragment)
            if width==390:
                go('results?section=benign')
                page.screenshot(path=str(args.output/'benign-results-mobile.png'),full_page=True)
            go('responses')
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'),width
            page.locator('tbody .row-title').first.click()
            expect(page.locator('#record-title')).to_be_visible(timeout=15000)
            assert page.evaluate('document.querySelector("dialog").scrollWidth <= document.querySelector("dialog").clientWidth'),width
            page.keyboard.press('Escape')
            if width==390:
                page.screenshot(path=str(args.output/'mobile.png'),full_page=True)
        checks.append('No page or dialog overflow at 320, 390, and 768 pixels')
        browser.close()
    result={'passed':not errors,'checks':checks,'browser_errors':errors}
    (args.output/'browser-checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    assert not errors,errors


if __name__=='__main__':
    main()
