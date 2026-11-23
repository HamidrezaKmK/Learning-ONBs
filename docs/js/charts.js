/* Small SVG chart kit for the experiments section: line charts (linear or log axes) with markers, std bands,
 * a legend that toggles series, and a crosshair tooltip that snaps to the nearest data position.
 * Data are the exact values extracted from the paper's figures (see figdata/*.json). */
(function () {
  'use strict';
  const NS = 'http://www.w3.org/2000/svg';
  const el = (tag, attrs, parent) => {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  };
  const h = (tag, cls, parent, text) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  };

  // series colors, validated for colour-vision deficiencies as two sets: {ours, fourier, pca} and {ours, rk1, rk2, rk4}
  const COL = { ours: '#a3243b', fourier: '#2f6fa3', pca: '#3d8b4f', grid: '#2f6fa3',
                rk1: '#c27a2c', rk2: '#7a5aa6', rk4: '#00968a', truth: '#1c1c1c' };

  function scale(type, d0, d1, r0, r1) {
    if (type === 'log') {
      const a = Math.log10(d0), b = Math.log10(d1);
      const f = v => r0 + (Math.log10(v) - a) / (b - a) * (r1 - r0);
      f.inv = p => Math.pow(10, a + (p - r0) / (r1 - r0) * (b - a));
      return f;
    }
    const f = v => r0 + (v - d0) / (d1 - d0) * (r1 - r0);
    f.inv = p => d0 + (p - r0) / (r1 - r0) * (d1 - d0);
    return f;
  }
  function logTicks(d0, d1) {
    const out = [];
    for (let e = Math.floor(Math.log10(d0)); e <= Math.ceil(Math.log10(d1)); e++) {
      const v = Math.pow(10, e);
      if (v >= d0 * 0.999 && v <= d1 * 1.001) out.push(v);
    }
    return out;
  }
  function linTicks(d0, d1, n) {
    const step0 = (d1 - d0) / (n || 5), mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= step0 * 0.999);
    const out = [];
    for (let v = Math.ceil(d0 / step - 1e-9) * step; v <= d1 + 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }
  const fmtLog = v => {
    const e = Math.round(Math.log10(v));
    if (Math.abs(Math.log10(v) - e) < 1e-6) return e === 0 ? '1' : e === 1 ? '10' : e === 2 ? '100' : '10' + sup(e);
    return String(v);
  };
  const SUP = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
  const sup = e => String(e).split('').map(c => SUP[c]).join('');
  function fmtNum(v) {
    if (v === null || v === undefined || isNaN(v)) return '–';
    const a = Math.abs(v);
    if (a !== 0 && (a < 1e-2 || a >= 1e4)) { const e = Math.floor(Math.log10(a)); return (v / Math.pow(10, e)).toFixed(2) + '×10' + sup(e); }
    return a >= 100 ? v.toFixed(0) : a >= 10 ? v.toFixed(1) : v.toFixed(3);
  }
  function marker(g, shape, x, y, color, r) {
    r = r || 4.2;
    if (shape === 'square') return el('rect', { x: x - r, y: y - r, width: 2 * r, height: 2 * r, fill: color, stroke: '#fff', 'stroke-width': 1.5 }, g);
    if (shape === 'triangle') return el('polygon', { points: `${x},${y - r * 1.25} ${x + r * 1.15},${y + r * 0.85} ${x - r * 1.15},${y + r * 0.85}`, fill: color, stroke: '#fff', 'stroke-width': 1.5 }, g);
    if (shape === 'diamond') return el('polygon', { points: `${x},${y - r * 1.3} ${x + r * 1.1},${y} ${x},${y + r * 1.3} ${x - r * 1.1},${y}`, fill: color, stroke: '#fff', 'stroke-width': 1.5 }, g);
    return el('circle', { cx: x, cy: y, r, fill: color, stroke: '#fff', 'stroke-width': 1.5 }, g);
  }
  const legendKey = (parent, s) => {
    const k = el('svg', { width: 22, height: 10, class: 'lk' }, parent);
    el('line', { x1: 1, y1: 5, x2: 21, y2: 5, stroke: s.color, 'stroke-width': 2.2, 'stroke-dasharray': s.dash || '' }, k);
    if (s.marker) marker(k, s.marker, 11, 5, s.color, 3.2);
    return k;
  };

  /* lineChart(container, opts)
   * opts: { x: {type, domain, label, ticks?, fmt?}, y: {...}, series: [{id, name, color, pts:[[x,y]..], dash, marker, width,
   *         band: [[x,lo,hi]..], hidden}], height, aspect, tooltip: true|false, snapX: [...] (x positions the crosshair snaps to),
   *         xFmt, yFmt, onHover(x), note }
   * returns { setCursor(x), update(seriesId, pts) } */
  function lineChart(container, o) {
    container.classList.add('chart');
    const legend = h('div', 'legend', container);
    const wrap = h('div', 'plot', container);
    const W = 520, H = o.height || 340, M = Object.assign({ l: 62, r: 14, t: 12, b: 50 }, o.margin || {});
    const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': o.title || '' }, wrap);
    const X = scale(o.x.type, o.x.domain[0], o.x.domain[1], M.l, W - M.r);
    const Y = scale(o.y.type, o.y.domain[0], o.y.domain[1], H - M.b, M.t);
    const clipId = 'c' + Math.random().toString(36).slice(2);
    el('rect', { x: M.l, y: M.t, width: W - M.l - M.r, height: H - M.t - M.b }, el('clipPath', { id: clipId }, el('defs', {}, svg)));
    // grid + axes
    const ax = el('g', { class: 'axis' }, svg);
    const xt = o.x.ticks || (o.x.type === 'log' ? logTicks(...o.x.domain) : linTicks(...o.x.domain, 6));
    const yt = o.y.ticks || (o.y.type === 'log' ? logTicks(...o.y.domain) : linTicks(...o.y.domain, 5));
    const xf = o.x.fmt || (o.x.type === 'log' ? fmtLog : v => String(v));
    const yf = o.y.fmt || (o.y.type === 'log' ? fmtLog : v => String(v));
    yt.forEach(v => {
      el('line', { x1: M.l, x2: W - M.r, y1: Y(v), y2: Y(v), class: 'gl' }, ax);
      el('text', { x: M.l - 8, y: Y(v) + 5, 'text-anchor': 'end' }, ax).textContent = yf(v);
    });
    xt.forEach(v => {
      el('line', { x1: X(v), x2: X(v), y1: H - M.b, y2: H - M.b + 5, class: 'tk' }, ax);
      el('text', { x: X(v), y: H - M.b + 21, 'text-anchor': 'middle' }, ax).textContent = xf(v);
    });
    el('line', { x1: M.l, x2: W - M.r, y1: H - M.b, y2: H - M.b, class: 'base' }, ax);
    if (o.x.label) el('text', { x: (M.l + W - M.r) / 2, y: H - 6, 'text-anchor': 'middle', class: 'al' }, ax).textContent = o.x.label;
    if (o.y.label) el('text', { x: 15, y: (M.t + H - M.b) / 2, 'text-anchor': 'middle', class: 'al', transform: `rotate(-90 15 ${(M.t + H - M.b) / 2})` }, ax).textContent = o.y.label;
    if (o.refY !== undefined) el('line', { x1: M.l, x2: W - M.r, y1: Y(o.refY), y2: Y(o.refY), class: 'ref' }, ax);

    const plot = el('g', { 'clip-path': `url(#${clipId})` }, svg);
    const S = {};
    const path = pts => { let d = '', pen = false;
      for (const p of pts) { if (!p || p[1] === null) { pen = false; continue; } d += (pen ? 'L' : 'M') + X(p[0]).toFixed(1) + ',' + Y(p[1]).toFixed(1); pen = true; }
      return d; };
    function draw(s) {
      if (S[s.id]) S[s.id].g.remove();
      const g = el('g', { class: 'series' }, plot);
      if (s.band) el('path', { d: path(s.band.map(b => [b[0], b[2]])) + path(s.band.slice().reverse().map(b => [b[0], b[1]])).replace('M', 'L') + 'Z',
                               fill: s.color, 'fill-opacity': 0.13, stroke: 'none' }, g);
      if (s.pts && s.pts.length) el('path', { d: path(s.pts), fill: 'none', stroke: s.color, 'stroke-width': s.width || 2, 'stroke-dasharray': s.dash || '',
                                    'stroke-linejoin': 'round', 'stroke-linecap': 'round', opacity: s.opacity || 1 }, g);
      if (s.marker && s.pts) s.pts.filter(p => p && p[1] !== null).forEach(p => marker(g, s.marker, X(p[0]), Y(p[1]), s.color, s.mr));
      if (s.dots) s.dots.forEach(p => el('circle', { cx: X(p[0]), cy: Y(p[1]), r: 2.6, fill: s.color }, g));
      g.style.display = s.hidden ? 'none' : '';
      S[s.id] = { g, s };
    }
    o.series.forEach(draw);

    // legend (toggles)
    o.series.filter(s => !s.noLegend).forEach(s => {
      const b = h('button', 'lg' + (s.hidden ? ' off' : ''), legend);
      b.type = 'button';
      b.setAttribute('aria-pressed', s.hidden ? 'false' : 'true');
      legendKey(b, s);
      h('span', '', b, s.name);
      b.addEventListener('click', () => {
        s.hidden = !s.hidden;
        b.classList.toggle('off', s.hidden);
        b.setAttribute('aria-pressed', s.hidden ? 'false' : 'true');
        S[s.id].g.style.display = s.hidden ? 'none' : '';
        if (cur !== null) setCursor(cur);
      });
    });
    if (o.note) h('div', 'chart-note', container, o.note);

    // crosshair + tooltip
    const hair = el('line', { y1: M.t, y2: H - M.b, class: 'hair' }, svg);
    const dots = el('g', {}, svg);
    const tip = h('div', 'tip', wrap);
    let cur = null;
    const snap = o.snapX || [...new Set(o.series.flatMap(s => (s.marker ? s.pts.map(p => p[0]) : [])))].sort((a, b) => a - b);
    function valueAt(s, x) {
      const P = (s.pts || []).filter(p => p && p[1] !== null);
      if (!P.length) return null;
      if (s.marker) { const p = P.find(q => Math.abs(q[0] - x) < 1e-9 * Math.max(1, Math.abs(x))); return p ? p[1] : null; }
      let i = 0;
      while (i < P.length - 1 && P[i + 1][0] < x) i++;
      if (x < P[0][0] || x > P[P.length - 1][0]) return null;
      const a = P[i], b = P[Math.min(i + 1, P.length - 1)];
      return b[0] === a[0] ? a[1] : a[1] + (b[1] - a[1]) * (x - a[0]) / (b[0] - a[0]);
    }
    function setCursor(x) {
      cur = x;
      if (x === null) { hair.style.display = 'none'; tip.style.display = 'none'; dots.innerHTML = ''; return; }
      const px = X(x);
      hair.style.display = ''; set(hair, { x1: px, x2: px });
      dots.innerHTML = '';
      tip.innerHTML = '';
      h('div', 'tip-h', tip, (o.x.name || '') + ' ' + (o.xFmt ? o.xFmt(x) : fmtNum(x)));
      o.series.filter(s => !s.hidden && !s.noTip).forEach(s => {
        const v = valueAt(s, x);
        if (v === null) return;
        if (v >= o.y.domain[0] && v <= o.y.domain[1]) el('circle', { cx: px, cy: Y(v), r: 4, fill: s.color, stroke: '#fff', 'stroke-width': 2 }, dots);
        const row = h('div', 'tip-r', tip);
        legendKey(row, s);
        h('b', '', row, o.yFmt ? o.yFmt(v) : fmtNum(v));
        h('span', '', row, s.name);
        if (s.band) {
          const b = s.band.reduce((best, q) => Math.abs(q[0] - x) < Math.abs(best[0] - x) ? q : best, s.band[0]);
          h('span', 'tip-band', row, '(' + fmtNum(b[1]) + '–' + fmtNum(b[2]) + ')');
        }
      });
      tip.style.display = 'block';
      const frac = (px - M.l) / (W - M.l - M.r);
      tip.style.left = (px / W * 100) + '%';
      tip.style.transform = frac > 0.55 ? 'translate(calc(-100% - 14px), 0)' : 'translate(14px, 0)';
    }
    function set(e, a) { for (const k in a) e.setAttribute(k, a[k]); }
    function nearest(x) {
      if (!snap.length) return x;
      return snap.reduce((b, v) => Math.abs(X(v) - X(x)) < Math.abs(X(b) - X(x)) ? v : b, snap[0]);
    }
    const toX = ev => {
      const r = svg.getBoundingClientRect();
      const px = (ev.clientX - r.left) / r.width * W;
      return X.inv(Math.max(M.l, Math.min(W - M.r, px)));
    };
    if (o.tooltip !== false) {
      svg.addEventListener('pointermove', ev => { const x = nearest(toX(ev)); setCursor(x); if (o.onHover) o.onHover(x); });
      svg.addEventListener('pointerleave', () => { if (o.sticky === undefined) { setCursor(null); if (o.onHover) o.onHover(null); } else setCursor(o.sticky()); });
    }
    setCursor(null);
    return {
      setCursor,
      update(id, pts) { const e = S[id]; if (!e) return; e.s.pts = pts; draw(e.s); },
      X, Y,
    };
  }

  // segmented control: options [{value, label}], onChange(value)
  function segmented(parent, label, options, value, onChange) {
    const w = h('div', 'seg', parent);
    if (label) h('span', 'seg-l', w, label);
    const btns = options.map(op => {
      const b = h('button', 'seg-b', w, op.label);
      b.type = 'button';
      b.addEventListener('click', () => select(op.value, true));
      return [op.value, b];
    });
    function select(v, fire) {
      btns.forEach(([val, b]) => { b.classList.toggle('on', val === v); b.setAttribute('aria-pressed', val === v ? 'true' : 'false'); });
      if (fire) onChange(v);
    }
    select(value, false);
    return { select };
  }

  window.Charts = { lineChart, segmented, COL, fmtNum, h, el };
})();
