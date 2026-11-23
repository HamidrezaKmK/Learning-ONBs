/* Bases4Spaces project page: wires the player to the chapter list and the per-keyframe discussion. */
(function () {
  'use strict';
  const FILES = ['01-space', '02-path', '03-step', '04-cayley', '05-frame', '06-integrate', '07-train'];
  const docs = {};

  // Markdown with LaTeX: math is cut out before Markdown runs, then typeset with KaTeX
  function renderMd(src) {
    const math = [];
    const keep = (t, display) => { math.push([t, display]); return '@@M' + (math.length - 1) + '@@'; };
    src = src.replace(/\$\$([\s\S]+?)\$\$/g, (_, t) => keep(t, true))
             .replace(/\$([^$\n]+?)\$/g, (_, t) => keep(t, false));
    let html = marked.parse(src);
    html = html.replace(/<p>@@M(\d+)@@<\/p>/g, (_, i) => '@@M' + i + '@@')
               .replace(/@@M(\d+)@@/g, (_, i) => katex.renderToString(math[+i][0], { displayMode: math[+i][1], throwOnError: false }));
    return html;
  }

  async function load() {
    await Promise.all(FILES.map(async (f, i) => {
      const r = await fetch('content/' + f + '.md');
      docs[i] = r.ok ? renderMd(await r.text()) : '<p>Missing <code>content/' + f + '.md</code>.</p>';
    }));
  }

  function init() {
    const root = document.getElementById('player');
    const P = window.B4SPlayer(root);
    window.b4s = P;
    const chapters = document.getElementById('chapters');
    const playBtn = document.getElementById('play');
    const disc = document.getElementById('discussion');
    const items = P.stages.map((s, i) => {
      const b = document.createElement('button');
      b.className = 'chap';
      b.innerHTML = '<span class="n">' + (i + 1) + '</span><span class="t">' + s.title + '</span><span class="bar"><i></i></span>';
      b.addEventListener('click', () => P.replay(i));
      chapters.appendChild(b);
      return b;
    });
    let shown = -1;
    P.on(({ stage, progress, T, playing }) => {
      items.forEach((b, i) => {
        b.classList.toggle('on', i === stage);
        const f = i < stage ? 1 : i === stage ? progress : 0;
        b.querySelector('i').style.width = (f * 100).toFixed(1) + '%';
      });
      playBtn.dataset.state = playing ? 'pause' : (T >= P.TOTAL - 1e-6 ? 'replay' : 'play');
      playBtn.setAttribute('aria-label', playing ? 'Pause' : 'Play to the next keyframe');
      if (stage !== shown && docs[stage] !== undefined) {
        shown = stage;
        disc.classList.remove('in');
        disc.innerHTML = '<div class="kicker">Keyframe ' + (stage + 1) + ' of ' + P.stages.length + '</div>' + docs[stage];
        void disc.offsetWidth; disc.classList.add('in');
      }
    });
    playBtn.addEventListener('click', () => P.play());
    document.getElementById('prev').addEventListener('click', () => P.prev());
    document.getElementById('next').addEventListener('click', () => P.next());
    root.querySelector('.stage').addEventListener('click', () => P.play());
    document.addEventListener('keydown', e => {
      if (e.target.closest('input,textarea,[contenteditable]')) return;
      if (e.key === 'ArrowRight') { e.preventDefault(); P.next(); }
      else if (e.key === 'ArrowLeft') { e.preventDefault(); P.prev(); }
      else if (e.key === ' ') { e.preventDefault(); P.play(); }
    });
    const stage = root.querySelector('.stage');
    new ResizeObserver(() => stage.style.setProperty('--u', (stage.clientWidth / 1280) + 'px')).observe(stage);
    const seek = /seek=([\d.]+)/.exec(location.hash);   // testing: #seek=<seconds> jumps the method animation there
    if (seek) { P.seek(+seek[1]); root.scrollIntoView(); return; }
    P.play();   // play up to the first keyframe
  }

  // typeset $...$ in the static sections (TL;DR, method intro, experiments)
  function typesetStatic() {
    if (!window.renderMathInElement) return;
    document.querySelectorAll('.sec, .sec-intro').forEach(el => renderMathInElement(el, {
      delimiters: [{ left: '$$', right: '$$', display: true }, { left: '$', right: '$', display: false }],
      throwOnError: false,
    }));
  }

  // inline SVG figures (an <img> SVG cannot load the tile images it references)
  document.querySelectorAll('[data-src]').forEach(el => {
    fetch(el.dataset.src).then(r => r.text()).then(t => {
      el.innerHTML = t;
      const svg = el.querySelector('svg');
      if (svg) { svg.setAttribute('role', 'img'); svg.setAttribute('aria-hidden', 'true'); }
    });
  });

  load().then(() => { typesetStatic(); init(); });
})();
