"""실험 노트를 정적 파일로 내보낸다 (GitHub Pages 용). 학습도 서버도 없이, 지금 있는 결과만 쓴다.

    py export_site.py           → docs/ (index.html · app.js · style.css · results.json)

결과가 바뀌면 다시 실행해 docs/ 를 함께 커밋한다.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from candidates.site import payload

ROOT = Path(__file__).resolve().parent
WEB, DOCS = ROOT / "web", ROOT / "docs"
APP_TAG = '<script src="app.js"></script>'


def main():
    DOCS.mkdir(exist_ok=True)
    (DOCS / "results.json").write_text(json.dumps(payload(), ensure_ascii=False), encoding="utf-8")
    for name in ("app.js", "style.css"):
        shutil.copyfile(WEB / name, DOCS / name)
    html = (WEB / "index.html").read_text(encoding="utf-8")
    if APP_TAG not in html:
        raise SystemExit(f"web/index.html 에 {APP_TAG} 가 없습니다")
    html = html.replace(APP_TAG, f"<script>window.RESULTS_URL = 'results.json';</script>\n{APP_TAG}")
    (DOCS / "index.html").write_text(html, encoding="utf-8")
    (DOCS / ".nojekyll").touch()               # Jekyll 변환 없이 파일을 그대로 내보낸다
    size = (DOCS / "results.json").stat().st_size
    print(f"내보냄: {DOCS} (results.json {size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
