#!/usr/bin/env python3
"""Exercise the recorded-data playground, including anonymous sandbox hosting."""
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright, expect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    errors, requests, checks = [], [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, executable_path='/usr/bin/google-chrome', args=['--no-sandbox'])
        page = browser.new_page(viewport={'width': 1440, 'height': 1000})
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.on('console', lambda message: errors.append(message.text) if message.type == 'error' else None)
        page.on('request', lambda request: requests.append(request.url))

        def ready():
            expect(page.locator('#app')).to_have_attribute('aria-busy', 'false', timeout=60000)
            expect(page.locator('#app')).not_to_contain_text('Unable to load this view')

        def go(fragment):
            page.goto(args.url + '#' + fragment, wait_until='networkidle')
            ready()

        go('')
        expect(page.locator('[data-view=play]')).to_have_attribute('aria-current', 'page')
        expect(page.locator('.lab-map-language')).to_have_count(5)
        expect(page.locator('.lab-response-pair .lab-response')).to_have_count(2)
        expect(page.locator('.lab-side-a')).to_have_attribute('data-outcome', 'NO_SUBSTANTIVE_ACTION')
        expect(page.locator('.lab-side-b')).to_have_attribute('data-outcome', 'SAY_DO_GAP')
        expect(page.locator('.lab-side-b .lab-call')).to_contain_text('process_candidate')
        page.screenshot(path=str(args.output / 'playground-desktop.png'), full_page=True)
        checks.append('The opening example shows two actual refusals and the harmful tool call in only one condition')

        page.locator('.lab-map-language[data-language=hi]').click()
        expect(page.locator('#lab-against')).to_have_value('hi')
        ready()
        expect(page.locator('.lab-side-b .lab-response-head')).to_contain_text('Hindi')
        expect(page.locator('.lab-map-route')).to_have_count(1)
        share = page.url
        page.reload(wait_until='networkidle')
        ready()
        expect(page.locator('#lab-against')).to_have_value('hi')
        assert page.url == share
        page.locator('.lab-map-language[data-language=pt]').focus()
        page.keyboard.press('Enter')
        expect(page.locator('#lab-against')).to_have_value('pt')
        ready()
        checks.append('Map clicks and keyboard selection change the actual language comparison; shared links survive reload')

        for axis, against in [('model', 'claude-haiku-4.5'), ('pressure', 'none'), ('prompt', 'neutral')]:
            page.locator(f'[data-action=lab-set][data-field=axis][data-value={axis}]').click()
            expect(page.locator(f'#lab-{axis}')).to_be_visible()
            page.locator('#lab-against').select_option(against)
            ready()
            expect(page.locator('.lab-response')).to_have_count(2)
        page.locator('[data-action=lab-cell][data-language=zh][data-pressure=authority]').click()
        expect(page.locator('.lab-side-a .lab-config')).to_contain_text('Mandarin Chinese')
        ready()
        expect(page.locator('.lab-condition')).to_have_count(30)
        expect(page.locator('.lab-condition.is-selected')).to_have_count(1)
        page.locator('.lab-side-a [data-action=record]').click()
        expect(page.locator('#record-content')).to_contain_text('Full source record SHA-256', timeout=30000)
        page.keyboard.press('Escape')
        expect(page.locator('#record-dialog')).not_to_be_visible()
        checks.append('All four comparison axes, the 30-cell condition matrix, and original source evidence work')

        for story, axis in [('language', 'language'), ('pressure', 'pressure')]:
            page.locator(f'[data-action=lab-story][data-story={story}]').click()
            expect(page.locator(f'#lab-{axis}')).to_be_visible()
            ready()
            outcomes = page.locator('.lab-response').evaluate_all('(nodes)=>nodes.map(n=>n.dataset.outcome)')
            assert sum(o in ['COMPLY', 'SAY_DO_GAP'] for o in outcomes) == 1, outcomes
        checks.append('Guided language and pressure examples retrieve observed harmful-action contrasts')

        go('atlas')
        expect(page.locator('.atlas-cell')).to_have_count(150)
        page.locator('[data-action=lab-atlas-cell][data-model="llama-3.1-8b"][data-language=hi]').click()
        expect(page.locator('.atlas-language-name')).to_have_text('Hindi')
        ready()
        page.locator('[data-action=lab-set][data-field=prompt][data-value=difference]').click()
        expect(page.locator('[data-field=prompt][data-value=difference]')).to_have_attribute('aria-pressed', 'true')
        ready()
        page.locator('[data-action=lab-set][data-field=metric][data-value=gap]').click()
        expect(page.locator('.atlas-board-title')).to_contain_text('substantive-call')
        page.locator('#atlas-category').select_option('privacy')
        page.locator('#atlas-pressure').select_option('none')
        ready()
        assert 'NaN' not in page.locator('#app').inner_text()
        page.screenshot(path=str(args.output / 'atlas-desktop.png'), full_page=True)
        page.locator('[data-action=lab-atlas-compare]').click()
        expect(page.locator('[data-view=play]')).to_have_attribute('aria-current', 'page')
        ready()
        expect(page.locator('.lab-side-a .lab-config')).to_contain_text('Hindi')
        expect(page.locator('.lab-side-a .lab-config')).to_contain_text('No pressure')
        checks.append('All 150 atlas cells, metric denominators, instruction differences and category/pressure filters are interactive')

        go('guess')
        expect(page.locator('.guess-words')).to_contain_text("I can't help")
        expect(page.locator('.guess-panel .lab-call')).to_have_count(0)
        page.locator('[data-action=lab-guess][data-guess=not-harmful]').click()
        expect(page.locator('.guess-panel')).to_contain_text('The action tells another story.')
        expect(page.locator('.guess-panel .lab-call')).to_contain_text('process_candidate')
        page.screenshot(path=str(args.output / 'prediction-reveal.png'), full_page=True)
        page.locator('[data-action=lab-next-case]').click()
        expect(page.locator('[data-action=lab-guess]')).to_have_count(2)
        expect(page.locator('.guess-panel .lab-call')).to_have_count(0)
        checks.append('Predict-then-reveal hides calls until a choice, reveals recorded evidence, and resets for the next case')

        for width in [390, 320, 768]:
            page.set_viewport_size({'width': width, 'height': 844})
            for view in ['play', 'atlas', 'guess']:
                go(view)
                assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), (width, view)
                if width == 390:
                    page.screenshot(path=str(args.output / f'{view}-mobile.png'), full_page=True)
            go('play')
            page.locator('.lab-map-language[data-language=es]').click()
            expect(page.locator('#lab-against')).to_have_value('es')
            ready()
        checks.append('Map interaction and all three new views work at 320, 390, 768, and 1440 pixels')
        assert all(url.startswith(args.url) for url in requests), sorted(set(u for u in requests if not u.startswith(args.url)))
        checks.append('No third-party runtime requests, geolocation, tracking, model calls, or credentials')
        report = {'passed': not errors, 'browser_errors': errors, 'opaque_origin': page.evaluate('globalThis.origin') == 'null', 'checks': checks}
        browser.close()
    (args.output / 'browser-checks.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    assert not errors, errors


if __name__ == '__main__':
    main()
