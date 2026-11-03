#!/usr/bin/env python3
"""Add gallery and GitHub links to published visualization HTML.

Run AFTER Hugo has generated public/, so HTML in inference-viz is never changed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import tempfile

REPO_URL = "https://github.com/yiliu30/inference-viz"
GALLERY_URL = "https://yiliu30.github.io/viz/"
MARKER = "<!-- inference-viz:github-navigation -->"
BODY_END = re.compile(r"</body\s*>", flags=re.IGNORECASE)

STYLE = """<style id="inference-viz-github-navigation-style">
/* Scoped to this small fixed bar; original page CSS remains untouched. */
#inference-viz-github-navigation {
  all: initial !important;
  position: fixed !important;
  right: max(12px, env(safe-area-inset-right)) !important;
  bottom: max(12px, env(safe-area-inset-bottom)) !important;
  z-index: 2147483645 !important;
  display: flex !important;
  align-items: center !important;
  gap: 6px !important;
  padding: 6px !important;
  box-sizing: border-box !important;
  max-width: calc(100vw - 24px) !important;
  border: 1px solid rgba(255,255,255,.25) !important;
  border-radius: 12px !important;
  background: rgba(17,24,39,.94) !important;
  box-shadow: 0 4px 16px rgba(0,0,0,.25) !important;
}
#inference-viz-github-navigation a {
  all: initial !important;
  display: inline-flex !important;
  box-sizing: border-box !important;
  align-items: center !important;
  justify-content: center !important;
  min-height: 30px !important;
  padding: 5px 9px !important;
  border-radius: 7px !important;
  color: #f9fafb !important;
  font: 600 12px/1.25 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif !important;
  text-decoration: none !important;
  white-space: nowrap !important;
  cursor: pointer !important;
  pointer-events: auto !important;
}
#inference-viz-github-navigation a:hover { background: #374151 !important; }
#inference-viz-github-navigation a:focus-visible {
  outline: 2px solid #7dd3fc !important;
  outline-offset: 2px !important;
}
#inference-viz-github-navigation a:last-child {
  background: #f9fafb !important;
  color: #111827 !important;
}
#inference-viz-github-navigation a:last-child:hover { background: #dbeafe !important; }
@media (max-width: 420px) {
  #inference-viz-github-navigation { gap: 2px !important; padding: 4px !important; }
  #inference-viz-github-navigation a { padding: 5px 7px !important; font-size: 11px !important; }
}
</style>"""


def bar(filename: str) -> str:
    return (
        f"{MARKER}\n{STYLE}\n"
        '<nav id="inference-viz-github-navigation" aria-label="Inference Viz project links">\n'
        f'  <a href="{GALLERY_URL}" aria-label="Back to visualization gallery">&#8592; Back to gallery</a>\n'
        f'  <a href="{REPO_URL}" target="_blank" rel="noopener noreferrer"'
        ' aria-label="Star Inference Viz on GitHub">&#9733; Star project</a>\n'
        '</nav>\n'
    )


def inject(html: str, filename: str) -> str:
    if MARKER in html:
        return html  # Safe to rerun without duplicating links.
    matches = list(BODY_END.finditer(html))
    if not matches:
        raise ValueError(f"{filename}: missing closing </body> tag")
    position = matches[-1].start()
    return html[:position] + bar(filename) + html[position:]


def publish(directory: Path) -> tuple[int, int]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing published visualization directory: {directory}")
    pages = [p for p in sorted(directory.glob("*.html")) if p.name != "admin.html"]
    if not pages:
        raise RuntimeError(f"No visualization pages found in {directory}")
    updated = 0
    for path in pages:
        original = path.read_text(encoding="utf-8")
        result = inject(original, path.name)
        if result != original:
            path.write_text(result, encoding="utf-8")
            updated += 1
    print(f"GitHub project navigation: {updated} updated, {len(pages)} visualization pages checked (admin excluded)")
    return updated, len(pages)


def self_test() -> None:
    sample = '<!doctype html>\n<html><head><title>Demo</title></head><body><button onclick="void(0)">Try</button></body></html>'
    first = inject(sample, 'tiny_gpu_trace_explorer (2).html')
    assert first.startswith(sample.split('</body>')[0])
    assert first.endswith('</body></html>')
    assert 'href="https://yiliu30.github.io/viz/"' in first
    assert 'Back to gallery' in first
    assert 'View source' not in first
    assert 'Star project' in first
    assert first.count(MARKER) == 1
    assert inject(first, 'tiny_gpu_trace_explorer (2).html') == first
    try:
        inject('<html><body>Broken', 'broken.html')
    except ValueError:
        pass
    else:
        raise AssertionError('A missing </body> tag must fail, not silently skip navigation')
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        (base / 'tiny_gpu_trace_explorer (2).html').write_text(sample, encoding='utf-8')
        (base / 'admin.html').write_text(sample, encoding='utf-8')
        assert publish(base) == (1, 1)
        assert publish(base) == (0, 1)
        assert (base / 'admin.html').read_text(encoding='utf-8') == sample
    print('Navigation injector self-test passed')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', nargs='?', type=Path, help='Published viz directory, e.g. public/viz')
    parser.add_argument('--self-test', action='store_true', help='Run a local smoke test')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        if not args.directory:
            parser.error('Provide the published visualization directory')
        publish(args.directory)


if __name__ == '__main__':
    main()
