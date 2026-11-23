# Project page

A static site for GitHub Pages: an SVG animation that stops at keyframes, with the math for each keyframe underneath.

- `index.html`, `style.css`: the page.
- `js/anim.js`: the animation, drawn as SVG and computed on the fly (no video, no frames in memory).
- `js/page.js`: the player controls and the discussion.
- `content/*.md`: one Markdown file per keyframe; `$...$` and `$$...$$` are typeset with KaTeX.

Preview locally (the page loads the Markdown with `fetch`, so it needs a server):

```
python -m http.server 8000 --directory docs
```

then open http://localhost:8000.

Publish: GitHub → Settings → Pages → Build and deployment → Deploy from a branch → `main`, folder `/docs`.
The page then lives at https://hamidrezakmk.github.io/Learning-ONBs/.

Keys: space plays to the next keyframe, ← and → step between keyframes.
