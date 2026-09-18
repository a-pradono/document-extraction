"""Render a Markdown document as a themed, self-contained HTML page.

    python -m eval.render_md report.md -o outputs/report.html

The report is the thing a reviewer reads end to end, and asking them to read raw
Markdown in a repository is a worse experience than the dashboard it accompanies. This
renders it with the dashboard's own stylesheet plus prose typography, so the two read as
one system, and links the two together.

No external assets: the CSS is inlined, so the page works over file:// and as an
artifact supporting file.
"""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path
from typing import Optional

import markdown

CSS = (Path(__file__).parent / "dashboard.css").read_text(encoding="utf-8")

PROSE = """
.prose{max-width:82ch;margin:0 auto;padding:2.5rem 1.5rem 5rem}
.prose h1{font-size:clamp(1.7rem,3.6vw,2.4rem);margin:0 0 .4rem;letter-spacing:-.03em}
.prose h2{font-size:1.25rem;margin:2.6rem 0 .8rem;padding-top:1.4rem;
  border-top:1px solid var(--line);letter-spacing:-.02em}
.prose h3{font-size:1.02rem;margin:1.9rem 0 .5rem;letter-spacing:-.01em}
.prose h4{font-size:.9rem;margin:1.3rem 0 .4rem;text-transform:none;letter-spacing:0;
  color:var(--fg);display:block}
.prose h4:after{content:none}
.prose p,.prose li{color:var(--mut);font-size:.95rem;line-height:1.7}
.prose strong{color:var(--fg)}
.prose blockquote{margin:1.4rem 0;padding:.9rem 1.1rem;background:var(--bg2);
  border-left:3px solid var(--accent);border-radius:0 10px 10px 0}
.prose blockquote p{margin:.3rem 0;color:var(--fg)}
.prose table{margin:1.2rem 0;font-size:.88rem}
.prose th{background:var(--bg2);white-space:nowrap}
.prose td,.prose th{border:1px solid var(--line);padding:.45rem .6rem;text-align:left;
  vertical-align:top}
.prose td:not(:first-child){font-variant-numeric:tabular-nums}
.prose code{font-family:var(--mono);font-size:.85em;background:var(--bg2);
  border:1px solid var(--line);border-radius:5px;padding:.05rem .3rem;color:var(--fg)}
.prose pre{background:var(--bg2);border:1px solid var(--line);border-radius:10px;
  padding:.9rem 1rem;overflow-x:auto}
.prose pre code{background:none;border:0;padding:0;font-size:.82rem;line-height:1.55}
.prose hr{border:0;border-top:1px solid var(--line);margin:2.2rem 0}
.prose a{color:var(--accent)}
.topbar{position:sticky;top:0;z-index:5;background:var(--bg);
  border-bottom:1px solid var(--line);padding:.7rem 1.5rem;display:flex;gap:.8rem;
  align-items:center;flex-wrap:wrap}
.topbar .name{font-weight:700;font-size:.9rem}
.topbar a{font-size:.82rem;color:var(--accent);text-decoration:none;font-weight:600}
.topbar a:hover{text-decoration:underline}
"""


def render(markdown_text: str, *, title: str, back_link: Optional[str] = None,
           back_label: str = "Dashboard") -> str:
    body = markdown.markdown(
        markdown_text,
        extensions=["tables", "fenced_code", "toc", "sane_lists", "attr_list"],
    )
    nav = (f'<a href="{html.escape(back_link)}">&larr; {html.escape(back_label)}</a>'
           if back_link else "")
    return (f'<!doctype html><html lang=en><head><meta charset=utf-8>'
            f'<meta name=viewport content="width=device-width,initial-scale=1">'
            f"<title>{html.escape(title)}</title><style>{CSS}{PROSE}</style></head><body>"
            f'<div class=topbar><span class=name>{html.escape(title)}</span>'
            f"<span class=spacer></span>{nav}</div>"
            f"<article class=prose>{body}</article></body></html>")


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path, help="the Markdown file")
    ap.add_argument("-o", "--out", type=Path, default=Path("outputs/report.html"))
    ap.add_argument("--title", default=None)
    ap.add_argument("--back", default="dashboard.html", help="link back to this page")
    args = ap.parse_args(argv)

    text = args.source.read_text(encoding="utf-8")
    heading = re.search(r"^#\s+(.+)$", text, re.M)
    title = args.title or (heading.group(1).strip() if heading else args.source.stem)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(text, title=title, back_link=args.back), encoding="utf-8")
    print(f"wrote {args.out} ({args.out.stat().st_size / 1000:.0f} kB)")


if __name__ == "__main__":
    main()
