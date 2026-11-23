/* Bases4Spaces: the keyframed explainer animation.
 * Everything is drawn as SVG (vector) and is a pure function of one time T, so seeking is instant and nothing is
 * cached per frame: the scene is built once and each frame only updates attributes. The functions on the cards are
 * real: Fourier sines rotated by rank-2 steps, computed in a 12-mode sine basis (exact inner products).
 * The loss landscape on the sphere is schematic (it illustrates the idea; it is not computed from data). */
(function () {
  'use strict';
  const NS = 'http://www.w3.org/2000/svg';
  const W = 1280, H = 720;
  const C = { ink: '#1c1c1c', muted: '#5d646b', red: '#750014', silver: '#8b959e', slate: '#4f6475',
              violet: '#6b4a87', rose: '#b0435c', alpha: '#2f6fa3', beta: '#c0702b', line: '#c6cdd4' };
  const FCOL = [C.slate, C.violet, C.rose, C.red];

  // ---------- small helpers ----------
  function el(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function set(e, attrs) { for (const k in attrs) e.setAttribute(k, attrs[k]); }
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  const ease = x => { x = clamp(x, 0, 1); return x * x * (3 - 2 * x); };
  const seg = (u, a, b) => ease((u - a) / (b - a));
  const lerp = (a, b, t) => a + (b - a) * t;
  function mixHex(a, b, t) {
    const pa = parseInt(a.slice(1), 16), pb = parseInt(b.slice(1), 16);
    const r = Math.round(lerp(pa >> 16, pb >> 16, t)), g = Math.round(lerp((pa >> 8) & 255, (pb >> 8) & 255, t)),
          bl = Math.round(lerp(pa & 255, pb & 255, t));
    return '#' + ((1 << 24) | (r << 16) | (g << 8) | bl).toString(16).slice(1);
  }
  const fmt = v => v.toFixed(1);
  const ptsStr = P => P.map(p => fmt(p[0]) + ',' + fmt(p[1])).join(' ');

  // ---------- function space: L2(0,1) through 12 sine modes ----------
  const NM = 12, NG = 120;
  const XS = Float64Array.from({ length: NG }, (_, i) => (i + 0.5) / NG);
  const SB = [];
  for (let m = 0; m < NM; m++) SB.push(Float64Array.from(XS, x => Math.SQRT2 * Math.sin((m + 1) * Math.PI * x)));
  function evalF(c) {
    const v = new Float64Array(NG);
    for (let m = 0; m < NM; m++) { const cm = c[m]; if (cm) for (let i = 0; i < NG; i++) v[i] += cm * SB[m][i]; }
    return v;
  }
  const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
  // exp(t K) f for K = alpha (x) beta - beta (x) alpha with alpha, beta orthonormal: rotate span{alpha, beta} by t
  function rot(a, b, f, t) {
    const ca = dot(a, f), cb = dot(b, f), c = Math.cos(t), s = Math.sin(t);
    const na = c * ca + s * cb, nb = -s * ca + c * cb, g = Float64Array.from(f);
    for (let m = 0; m < NM; m++) g[m] += (na - ca) * a[m] + (nb - cb) * b[m];
    return g;
  }
  function rng(seed) {
    return function () {
      seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  const gauss = r => Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(2 * Math.PI * r());
  function plane(seed, decay) {
    const r = rng(seed);
    let a = new Float64Array(NM), b = new Float64Array(NM);
    for (let m = 0; m < NM; m++) { a[m] = gauss(r) * Math.pow(decay, m); b[m] = gauss(r) * Math.pow(decay, m); }
    const na = Math.sqrt(dot(a, a)); a = a.map(x => x / na);
    const p = dot(a, b); b = b.map((x, m) => x - p * a[m]);
    const nb = Math.sqrt(dot(b, b)); b = b.map(x => x / nb);
    return [a, b];
  }
  const NCARD = 5;
  const E0 = [...Array(NCARD)].map((_, k) => { const c = new Float64Array(NM); c[k] = 1; return c; });

  // the plane of the zoomed step: chosen so the eight arrows (four functions, before and after) stay apart
  const DT = 0.7;
  function pickPlane() {
    let best = null, bestScore = -1;
    for (let s = 1; s < 400; s++) {
      const [a, b] = plane(s, 0.62);
      const angs = [], lens = [];
      for (let k = 0; k < 4; k++) { const th = Math.atan2(b[k], a[k]); angs.push(th, th - DT); lens.push(Math.hypot(a[k], b[k])); }
      if (Math.min(...lens) < 0.33) continue;
      let sep = 9;
      for (let i = 0; i < angs.length; i++) for (let j = i + 1; j < angs.length; j++) {
        let d = Math.abs(angs[i] - angs[j]) % (2 * Math.PI); d = Math.min(d, 2 * Math.PI - d); sep = Math.min(sep, d);
      }
      if (sep > bestScore) { bestScore = sep; best = [a, b]; }
    }
    return best;
  }
  const [A0, B0] = pickPlane();

  // the path: NW rank-2 steps (the first is the zoomed one); training blends random planes into these
  const NW = 24, THW = 0.3;
  const PL = [[A0, B0]], RP = [];
  for (let j = 1; j < NW; j++) PL.push(plane(100 + j, 0.78));
  for (let j = 0; j < NW; j++) RP.push(plane(500 + j, 0.78));
  function basisAt(s, k) {
    const x = clamp(s, 0, 1) * NW, J = Math.min(Math.floor(x), NW - 1);
    let f = E0[k];
    for (let j = 0; j <= J; j++) {
      const w = j < J ? 1 : x - J;
      if (w > 0) f = rot(PL[j][0], PL[j][1], f, THW * w);
    }
    return f;
  }
  function frameTrain(lam, k) {
    let f = E0[k];
    for (let j = 0; j < NW; j++) { f = rot(RP[j][0], RP[j][1], f, (1 - lam) * THW); f = rot(PL[j][0], PL[j][1], f, lam * THW); }
    return f;
  }

  // ---------- the sphere: a cartoon of the space of orthonormal bases ----------
  const SC = [500, 365], R = 250, SR = 2.95, ELV = Math.asin(0.55 / 2.95), CE = Math.cos(ELV), SE = Math.sin(ELV);
  const sproj = v => [SC[0] + R * v[0], SC[1] - R * (v[1] * CE - v[2] * SE), v[2] * CE + v[1] * SE];
  function liftN(X, Y) {
    const x = clamp(X, -1, 1), r = Math.sqrt(Math.max(1 - x * x, 1e-9)), q = clamp(Y / r, -1, 1), ph = Math.acos(q) - ELV;
    return [x, r * Math.cos(ph), r * Math.sin(ph)];
  }
  const lift = p => liftN(p[0] / SR, p[1] / SR);
  const norm3 = v => { const n = Math.hypot(v[0], v[1], v[2]); return [v[0] / n, v[1] / n, v[2] / n]; };
  const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  // a square of half side `half` in the plane tangent to the sphere at rel, projected like the sphere
  function tangentPatch(rel, half) {
    const n = lift(rel), u = norm3(cross([0, 1, 0], n)), v = cross(n, u);
    return [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([a, b]) => sproj([0, 1, 2].map(q => n[q] + half * (a * u[q] + b * v[q]))));
  }
  function spherePath(rel, n = 16) {
    const P = rel.map(lift), Pp = [P[0], ...P, P[P.length - 1]], out = [];
    for (let i = 1; i < Pp.length - 2; i++) {
      const p0 = Pp[i - 1], p1 = Pp[i], p2 = Pp[i + 1], p3 = Pp[i + 2];
      for (let s = 0; s < n; s++) {
        const t = s / n;
        const q = [0, 1, 2].map(d => 0.5 * (2 * p1[d] + (-p0[d] + p2[d]) * t + (2 * p0[d] - 5 * p1[d] + 4 * p2[d] - p3[d]) * t * t
                                           + (-p0[d] + 3 * p1[d] - 3 * p2[d] + p3[d]) * t * t * t));
        out.push(sproj(norm3(q)));
      }
    }
    out.push(sproj(P[P.length - 1]));
    return out;
  }
  const REL = [[1.45, 1.75], [0.35, 2.0], [-0.85, 1.6], [-1.25, 0.75], [-0.55, 0.1], [0.75, -0.05], [1.3, -0.8], [0.45, -1.55], [-0.55, -1.2], [-1.55, -1.75]];
  // a random initial path: REL nudged sideways by a smooth bump, so every blend with REL is a simple curve
  const REL0 = [[1.45, 1.75], [0.37, 1.86], [-0.58, 1.31], [-0.56, 0.85], [-0.01, 0.72], [1.56, 0.54], [2.65, -0.95], [1.7, -1.96], [1.1, -1.41], [0.65, -1.85]];
  const PATH = spherePath(REL);
  // schematic loss: geodesic ellipses around the learned basis (the end of the path)
  const VL = lift(REL[REL.length - 1]);
  const U1 = norm3([1 - VL[0] * VL[0], -VL[0] * VL[1], -VL[0] * VL[2]]), U2 = cross(VL, U1);
  const ECC = 0.42;
  function lossAt(v) {
    const rho = Math.acos(clamp(dot3(v, VL), -1, 1)), ph = Math.atan2(dot3(v, U2), dot3(v, U1));
    return rho * Math.sqrt(Math.cos(ph) ** 2 + Math.sin(ph) ** 2 / ECC);
  }
  const LEVELS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25];
  const LOWC = '#efc9d1', HIGHC = '#dfe6ec';
  const lossCol = L => mixHex(LOWC, HIGHC, clamp((L - 0.1) / 2.1, 0, 1));
  const levelCol = i => mixHex('#8d2a40', '#a2abb4', i / (LEVELS.length - 1));

  // ---------- scene ----------
  let svg, labelLayer, built = false;
  const S = {};          // persistent elements
  const LBL = [];

  function inlineTex(s) {
    // "text $math$ text" -> HTML with KaTeX
    return s.split(/(\$[^$]+\$)/g).map(p => p.startsWith('$') && p.endsWith('$') && p.length > 1
      ? katex.renderToString(p.slice(1, -1), { throwOnError: false }) : p.replace(/&/g, '&amp;').replace(/</g, '&lt;')).join('');
  }
  function label(src, { size = 20, color = C.ink, anchor = 'c', cls = '' } = {}) {
    const d = document.createElement('div');
    d.className = 'lbl ' + cls;
    d.style.color = color;
    d.style.fontSize = 'calc(var(--u) * ' + size + ')';
    d.innerHTML = inlineTex(src);
    labelLayer.appendChild(d);
    const tx = anchor === 'l' ? '0%' : anchor === 'r' ? '-100%' : '-50%';
    const o = { d, last: '', at(x, y, op) {
      const key = fmt(x) + ',' + fmt(y) + ',' + op.toFixed(2);
      if (key === o.last) return; o.last = key;
      if (op < 0.01) { d.style.visibility = 'hidden'; return; }
      d.style.visibility = 'visible';
      d.style.opacity = op;
      d.style.left = (x / W * 100) + '%';
      d.style.top = (y / H * 100) + '%';
      d.style.transform = 'translate(' + tx + ',-50%)';
    } };
    LBL.push(o);
    return o;
  }
  function arrow(parent, col, width) {
    const g = el('g', {}, parent);
    const ln = el('line', { stroke: col, 'stroke-width': width, 'stroke-linecap': 'round' }, g);
    const hd = el('polygon', { fill: col }, g);
    return {
      g, set(x1, y1, x2, y2, op, color) {
        const L = Math.hypot(x2 - x1, y2 - y1);
        if (op < 0.01 || L < 1) { g.setAttribute('opacity', 0); return; }
        const ux = (x2 - x1) / L, uy = (y2 - y1) / L, hl = Math.min(14, L * 0.4), hw = hl * 0.45;
        set(g, { opacity: op });
        if (color) { ln.setAttribute('stroke', color); hd.setAttribute('fill', color); }
        set(ln, { x1: fmt(x1), y1: fmt(y1), x2: fmt(x2 - ux * hl * 0.8), y2: fmt(y2 - uy * hl * 0.8) });
        hd.setAttribute('points', ptsStr([[x2, y2], [x2 - ux * hl - uy * hw, y2 - uy * hl + ux * hw], [x2 - ux * hl + uy * hw, y2 - uy * hl - ux * hw]]));
      }
    };
  }
  function curvePts(vals, x0, x1, y0, amp) {
    const out = [];
    for (let i = 0; i < NG; i += 2) out.push([x0 + (x1 - x0) * i / (NG - 1), y0 - amp * vals[i]]);
    return out;
  }
  // a stack of cards, one function per card (front card = first function)
  function Stack(parent, n = NCARD) {
    const g = el('g', {}, parent), cards = [];
    for (let k = n - 1; k >= 0; k--) {
      const cg = el('g', {}, g);
      cards[k] = { cg, r: el('rect', { rx: 5, fill: '#ffffff', 'stroke-width': 1.3 }, cg),
                   p: el('polyline', { fill: 'none', 'stroke-width': 1.7, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' }, cg) };
    }
    return {
      g, update({ x, y, s = 1, funcs, col, op = 1, w = 84, h = 58, dx = 12, dy = -9 }) {
        if (op < 0.01) { g.setAttribute('opacity', 0); return; }
        g.setAttribute('opacity', op);
        for (let k = 0; k < n; k++) {
          const c = cards[k], ox = x + k * dx * s, oy = y + k * dy * s;
          const fade = 1 - k * 0.14;
          set(c.r, { x: fmt(ox), y: fmt(oy), width: fmt(w * s), height: fmt(h * s), stroke: col, 'stroke-opacity': fade });
          set(c.p, { stroke: col, 'stroke-opacity': fade,
                     points: ptsStr(curvePts(evalF(funcs[k]), ox + 6 * s, ox + (w - 6) * s, oy + h * s / 2, h * s * 0.17)) });
        }
      }
    };
  }
  function Card(parent) {
    const g = el('g', {}, parent);
    const r = el('rect', { rx: 6, fill: '#ffffff', 'stroke-width': 1.3 }, g);
    const ghost = el('polyline', { fill: 'none', stroke: '#9aa3ab', 'stroke-width': 1.6, 'stroke-dasharray': '4 4' }, g);
    const p = el('polyline', { fill: 'none', 'stroke-width': 2.2, 'stroke-linejoin': 'round' }, g);
    return {
      g, update({ x, y, w, h, f, ghostF = null, col, op = 1, sw = 1.3, ghostOp = 1 }) {
        if (op < 0.01) { g.setAttribute('opacity', 0); return; }
        set(g, { opacity: op });
        set(r, { x: fmt(x), y: fmt(y), width: fmt(w), height: fmt(h), stroke: col, 'stroke-width': sw });
        set(p, { stroke: col, points: ptsStr(curvePts(evalF(f), x + 8, x + w - 8, y + h / 2, h * 0.17)) });
        if (ghostF) set(ghost, { opacity: ghostOp, points: ptsStr(curvePts(evalF(ghostF), x + 8, x + w - 8, y + h / 2, h * 0.17)) });
        else ghost.setAttribute('opacity', 0);
      }
    };
  }

  function build(root) {
    svg = root.querySelector('svg');
    labelLayer = root.querySelector('.labels');
    const defs = el('defs', {}, svg);
    const clip = el('clipPath', { id: 'b4s-clip' }, defs);
    el('circle', { cx: SC[0], cy: SC[1], r: R }, clip);
    const lit = el('radialGradient', { id: 'b4s-lit', cx: '0.36', cy: '0.3', r: '0.78' }, defs);
    el('stop', { offset: '0', 'stop-color': '#ffffff', 'stop-opacity': 0.75 }, lit);
    el('stop', { offset: '0.45', 'stop-color': '#ffffff', 'stop-opacity': 0.08 }, lit);
    el('stop', { offset: '1', 'stop-color': '#5b6874', 'stop-opacity': 0.30 }, lit);
    const sh = el('radialGradient', { id: 'b4s-shadow' }, defs);
    el('stop', { offset: '0', 'stop-color': '#5b6874', 'stop-opacity': 0.28 }, sh);
    el('stop', { offset: '1', 'stop-color': '#5b6874', 'stop-opacity': 0 }, sh);
    const lg = el('linearGradient', { id: 'b4s-legend' }, defs);
    el('stop', { offset: '0', 'stop-color': LOWC }, lg);
    el('stop', { offset: '1', 'stop-color': HIGHC }, lg);

    // world: everything that lives on the sphere (zooms as one group)
    S.world = el('g', {}, svg);
    el('ellipse', { cx: SC[0], cy: SC[1] + R + 26, rx: R * 0.82, ry: 22, fill: 'url(#b4s-shadow)' }, S.world);
    const shade = el('g', { 'clip-path': 'url(#b4s-clip)' }, S.world);
    el('circle', { cx: SC[0], cy: SC[1], r: R, fill: HIGHC }, shade);
    // loss shading: one colour per 12-unit cell, painted once into a tiny canvas and smoothed by bilinear upscaling,
    // then shown as a single image (cheap to zoom; an SVG blur filter over many cells stalls phones during the zoom)
    const STEP = 12, x0 = SC[0] - R - STEP, y0 = SC[1] - R - STEP, NC = Math.ceil((2 * R + 2 * STEP) / STEP);
    const small = document.createElement('canvas'); small.width = small.height = NC;
    const sctx = small.getContext('2d'), px = sctx.createImageData(NC, NC);
    for (let j = 0; j < NC; j++) for (let i = 0; i < NC; i++) {
      const X = (x0 + (i + 0.5) * STEP - SC[0]) / R, Y = (SC[1] - (y0 + (j + 0.5) * STEP)) / R;
      const hex = X * X + Y * Y > 1.05 ? HIGHC : lossCol(lossAt(liftN(X, Y))), v = parseInt(hex.slice(1), 16), o = 4 * (j * NC + i);
      px.data[o] = v >> 16; px.data[o + 1] = (v >> 8) & 255; px.data[o + 2] = v & 255; px.data[o + 3] = 255;
    }
    sctx.putImageData(px, 0, 0);
    const big = document.createElement('canvas'); big.width = big.height = NC * 8;
    const bctx = big.getContext('2d'); bctx.imageSmoothingEnabled = true; bctx.imageSmoothingQuality = 'high';
    bctx.drawImage(small, 0, 0, NC * 8, NC * 8);
    el('image', { href: big.toDataURL('image/png'), x: x0, y: y0, width: NC * STEP, height: NC * STEP, preserveAspectRatio: 'none' }, shade);
    el('circle', { cx: SC[0], cy: SC[1], r: R, fill: 'url(#b4s-lit)' }, shade);
    // graticule (front only, faint)
    const grat = el('g', { fill: 'none', stroke: '#ffffff', 'stroke-opacity': 0.7, 'stroke-width': 1 }, S.world);
    for (const lat of [-0.5, 0, 0.5]) {
      const P = [];
      for (let i = 0; i <= 120; i++) { const a = i / 120 * 2 * Math.PI, c = Math.cos(lat); P.push(sproj([c * Math.cos(a), Math.sin(lat), c * Math.sin(a)])); }
      el('path', { d: frontPath(P) }, grat);
    }
    for (const lon of [-1.0, 0, 1.0]) {
      const P = [];
      for (let i = 0; i <= 120; i++) { const a = -Math.PI / 2 + i / 120 * Math.PI; P.push(sproj([Math.cos(a) * Math.sin(lon), Math.sin(a), Math.cos(a) * Math.cos(lon)])); }
      el('path', { d: frontPath(P) }, grat);
    }
    el('circle', { cx: SC[0], cy: SC[1], r: R, fill: 'none', stroke: C.silver, 'stroke-width': 1.6 }, S.world);
    // level sets of the (schematic) loss
    S.levels = el('g', { fill: 'none' }, S.world);
    LEVELS.forEach((c, i) => {
      const P = [];
      for (let j = 0; j <= 180; j++) {
        const ph = j / 180 * 2 * Math.PI, rho = c / Math.sqrt(Math.cos(ph) ** 2 + Math.sin(ph) ** 2 / ECC);
        if (rho >= Math.PI) { P.push(null); continue; }
        const d = [0, 1, 2].map(q => Math.cos(rho) * VL[q] + Math.sin(rho) * (Math.cos(ph) * U1[q] + Math.sin(ph) * U2[q]));
        P.push(sproj(d));
      }
      el('path', { d: frontPath(P), stroke: levelCol(i), 'stroke-width': 1.5, 'stroke-opacity': 0.85 }, S.levels);
      el('path', { d: frontPath(P, true), stroke: levelCol(i), 'stroke-width': 1.1, 'stroke-opacity': 0.3, 'stroke-dasharray': '4 5' }, S.levels);
    });
    // paths
    S.ghosts = [];
    for (let i = 0; i < 6; i++) S.ghosts.push(el('polyline', { fill: 'none', stroke: C.silver, 'stroke-width': 2, 'stroke-linecap': 'round' }, S.world));
    S.pathGlow = el('polyline', { fill: 'none', stroke: C.red, 'stroke-opacity': 0.14, 'stroke-width': 10, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }, S.world);
    S.path = el('polyline', { fill: 'none', stroke: C.red, 'stroke-width': 3.2, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }, S.world);
    S.pathT = el('polyline', { fill: 'none', stroke: C.red, 'stroke-width': 3.2, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }, S.world);
    // markers
    S.pS = PATH[0]; S.pX = PATH[PATH.length - 1]; S.pR = sproj(lift([2.3, -1.2]));
    S.mStart = el('rect', { x: S.pS[0] - 7, y: S.pS[1] - 7, width: 14, height: 14, rx: 2, fill: C.slate, stroke: '#fff', 'stroke-width': 2 }, S.world);
    S.mRand = el('circle', { cx: S.pR[0], cy: S.pR[1], r: 7, fill: C.silver, stroke: '#fff', 'stroke-width': 2 }, S.world);
    S.mOpt = el('g', {}, S.world);
    el('path', { d: `M${S.pX[0] - 8},${S.pX[1] - 8}L${S.pX[0] + 8},${S.pX[1] + 8}M${S.pX[0] + 8},${S.pX[1] - 8}L${S.pX[0] - 8},${S.pX[1] + 8}`,
                 stroke: C.red, 'stroke-width': 4, 'stroke-linecap': 'round' }, S.mOpt);
    S.mEnd = el('circle', { r: 6, fill: C.red, stroke: '#fff', 'stroke-width': 2 }, S.world);

    // overlays outside the world group
    S.legend = el('g', {}, svg);
    el('rect', { x: 880, y: 108, width: 220, height: 12, rx: 6, fill: 'url(#b4s-legend)', stroke: C.line }, S.legend);
    S.connX = el('line', { stroke: C.silver, 'stroke-width': 1.4, 'stroke-dasharray': '4 4' }, svg);
    S.stR = Stack(svg); S.stF = Stack(svg); S.stX = Stack(svg); S.stRide = Stack(svg); S.stT = Stack(svg);

    // tangent space at the zoomed point: a patch of T_Q SO(inf) on the sphere that the zoom opens into the panel's backdrop
    S.tpatch = tangentPatch(REL[0], 0.14);
    S.tplane = el('polygon', { fill: '#f4f1f6', stroke: C.violet, 'stroke-opacity': 0.5, 'stroke-width': 1.6, 'stroke-linejoin': 'round' }, svg);
    S.tdot = el('circle', { r: 5, fill: C.violet }, svg);

    // zoomed panel: one step
    S.panel = el('g', {}, svg);
    S.top = [0, 1, 2, 3].map(() => Card(S.panel));
    S.bot = [0, 1, 2, 3].map(() => Card(S.panel));
    S.rows = el('g', {}, S.panel);
    const mkRow = (y, col) => {
      const g = el('g', {}, S.rows);
      el('line', { x1: 110, y1: y, x2: 740, y2: y, stroke: C.line, 'stroke-width': 1.2 }, g);
      const vals = evalF(col === C.alpha ? A0 : B0), mx = Math.max(...vals.map(Math.abs));
      el('polyline', { points: ptsStr(curvePts(vals.map(v => v / mx), 110, 740, y, 34)), fill: 'none', stroke: col, 'stroke-width': 2.4 }, g);
      for (let q = 0; q < 12; q++) {
        const i = Math.round((q + 0.5) / 12 * (NG - 1)), x = 110 + 630 * i / (NG - 1), yy = y - 34 * vals[i] / mx;
        el('line', { x1: x, y1: y, x2: x, y2: yy, stroke: col, 'stroke-opacity': 0.55, 'stroke-width': 1.4 }, g);
        el('circle', { cx: x, cy: yy, r: 3.2, fill: col }, g);
      }
      return g;
    };
    S.rowA = mkRow(318, C.alpha); S.rowB = mkRow(418, C.beta);
    S.IC = [1030, 395]; S.RAD = 150;
    S.circ = el('g', {}, S.panel);
    el('circle', { cx: S.IC[0], cy: S.IC[1], r: S.RAD, fill: '#f6f7f8', stroke: C.line, 'stroke-width': 1.4 }, S.circ);
    arrow(S.circ, C.alpha, 2.2).set(S.IC[0] - S.RAD - 18, S.IC[1], S.IC[0] + S.RAD + 22, S.IC[1], 1);
    arrow(S.circ, C.beta, 2.2).set(S.IC[0], S.IC[1] + S.RAD + 18, S.IC[0], S.IC[1] - S.RAD - 22, 1);
    S.ghostArr = [0, 1, 2, 3].map(() => arrow(S.panel, '#a2abb4', 2.6));
    S.arcs = [0, 1, 2, 3].map(k => el('path', { fill: 'none', stroke: FCOL[k], 'stroke-width': 2, 'stroke-opacity': 0.8 }, S.panel));
    S.arr = [0, 1, 2, 3].map(k => arrow(S.panel, FCOL[k], 3.4));
    S.links = [0, 1, 2, 3].map(() => arrow(S.panel, '#b8c0c7', 1.6));

    // frame + Gram
    S.fr = el('g', {}, svg);
    S.stBefore = Stack(S.fr); S.stAfter = Stack(S.fr);
    S.frArrow = arrow(S.fr, C.silver, 2);
    S.gram = el('g', {}, svg);
    S.gcells = [];
    const G0 = [830, 205], GC = 64;
    for (let i = 0; i < NCARD; i++) for (let j = 0; j < NCARD; j++) {
      const r = el('rect', { x: G0[0] + j * GC, y: G0[1] + i * GC, width: GC - 4, height: GC - 4, rx: 6, fill: i === j ? '#f1dde2' : '#f1f3f5' }, S.gram);
      const t = el('text', { x: G0[0] + j * GC + (GC - 4) / 2, y: G0[1] + i * GC + (GC - 4) / 2 + 6, 'text-anchor': 'middle', 'font-size': 17,
                             fill: i === j ? C.red : C.muted, 'font-family': 'Source Sans 3, sans-serif' }, S.gram);
      S.gcells.push({ i, j, t, r });
    }
    // loss plot
    S.plot = el('g', {}, svg);
    S.LO = [880, 470]; S.LW = 300; S.LH = 230;
    arrow(S.plot, C.silver, 1.8).set(S.LO[0], S.LO[1], S.LO[0] + S.LW, S.LO[1], 1);
    arrow(S.plot, C.silver, 1.8).set(S.LO[0], S.LO[1], S.LO[0], S.LO[1] - S.LH, 1);
    S.lcurve = el('polyline', { fill: 'none', stroke: C.red, 'stroke-width': 2.6 }, S.plot);
    S.ldot = el('circle', { r: 5, fill: C.red }, S.plot);

    // labels
    const L = S.L = {};
    L.so = label('$\\mathrm{SO}(\\infty)$', { size: 24, color: C.muted });
    L.legLo = label('low $\\mathcal{L}$', { size: 15, color: C.muted, anchor: 'r' });
    L.legHi = label('high $\\mathcal{L}$', { size: 15, color: C.muted, anchor: 'l' });
    L.legT = label('loss on bases (schematic)', { size: 15, color: C.muted });
    L.rand = label('a random basis', { size: 17, color: C.muted });
    L.four = label('Fourier basis', { size: 17, color: C.slate, anchor: 'r' });
    L.opt = label('the optimum', { size: 17, color: C.red });
    L.learned = label('learned basis', { size: 17, color: C.red });
    L.ode = label('$\\tfrac{d}{dt}Q(t)=K_\\theta(t)\\,Q(t),\\quad Q(0)=I$', { size: 22 });
    L.ode2 = label('$K_\\theta(t)=\\alpha_\\theta^t\\otimes\\beta_\\theta^t-\\beta_\\theta^t\\otimes\\alpha_\\theta^t$', { size: 20, color: C.muted });
    L.head = label('$e_k^{t+\\Delta t}=\\exp\\!\\big(\\Delta t\\,(\\alpha_\\theta^t\\otimes\\beta_\\theta^t-\\beta_\\theta^t\\otimes\\alpha_\\theta^t)\\big)\\,e_k^{t}$', { size: 24 });
    L.topT = label('basis at time $t$', { size: 15, color: C.muted, anchor: 'l' });
    L.botT = label('basis at time $t+\\Delta t$', { size: 15, color: C.muted, anchor: 'l' });
    L.a = label('$\\alpha_\\theta^t$', { size: 21, color: C.alpha, anchor: 'r' });
    L.b = label('$\\beta_\\theta^t$', { size: 21, color: C.beta, anchor: 'r' });
    L.q = label('evaluated at quadrature points $x_1,\\dots,x_N$', { size: 14, color: C.muted });
    L.ptitle = label('projection onto $\\mathrm{span}\\{\\alpha_\\theta^t,\\beta_\\theta^t\\}$', { size: 17, color: C.muted });
    L.pa = label('$\\alpha_\\theta^t$', { size: 18, color: C.alpha, anchor: 'l' });
    L.pb = label('$\\beta_\\theta^t$', { size: 18, color: C.beta });
    L.pnote = label('turns this plane by $\\Delta t$, fixes everything orthogonal to it', { size: 15, color: C.muted });
    L.ek = [0, 1, 2, 3].map(k => label('$e_' + (k + 1) + '^t$', { size: 17, color: FCOL[k] }));
    L.en = [0, 1, 2, 3].map(k => label('$e_' + (k + 1) + '^{t+\\Delta t}$', { size: 17, color: FCOL[k] }));
    L.dt = label('$\\Delta t$', { size: 16, color: C.muted });
    L.tan = label('tangent space $T_{Q(t)}\\,\\mathrm{SO}(\\infty)$', { size: 15, color: C.violet, anchor: 'r' });
    L.frA = label('frame at $t$', { size: 17, color: C.slate });
    L.frB = label('frame at $t+\\Delta t$', { size: 17, color: C.violet });
    L.frOp = label('one operator for every $e_k$', { size: 15, color: C.muted });
    L.gT = label('Gram matrix $\\langle e_i^{t},e_j^{t}\\rangle$, computed live', { size: 17, color: C.muted });
    L.gB = label('$\\langle Ce_i,\\,Ce_j\\rangle=\\langle e_i,\\,C^{*}C\\,e_j\\rangle=\\delta_{ij}$', { size: 19 });
    L.th0 = label('$\\theta^{(0)}$', { size: 18, color: C.muted, anchor: 'r' });
    L.ly = label('$\\mathcal{L}(\\theta)$', { size: 19, anchor: 'r' });
    L.lx = label('gradient steps', { size: 15, color: C.muted });
    L.lnote = label('schematic curve', { size: 13, color: C.muted, anchor: 'r' });
    L.rule = label('$\\theta\\leftarrow\\theta-\\eta\\,\\nabla_\\theta\\mathcal{L}(\\theta)$', { size: 21 });
    built = true;
  }
  // SVG path through projected points, keeping only the front (or only the back) side
  function frontPath(P, back = false) {
    let d = '', pen = false;
    for (const p of P) {
      const vis = p && (back ? p[2] < 0 : p[2] >= 0);
      if (!vis) { pen = false; continue; }
      d += (pen ? 'L' : 'M') + fmt(p[0]) + ',' + fmt(p[1]);
      pen = true;
    }
    return d || 'M0,0';
  }

  // ---------- timeline ----------
  const STAGES = [
    { id: 'space', title: 'The space of bases', dur: 7 },
    { id: 'path', title: 'A path of rotations', dur: 6 },
    { id: 'zoom', title: 'One small step', dur: 6 },
    { id: 'steps', title: 'Every function turns', dur: 11 },
    { id: 'frame', title: 'The whole frame', dur: 7 },
    { id: 'ride', title: 'Integrating the path', dur: 7 },
    { id: 'train', title: 'Learning the path', dur: 10 },
  ];
  const CUM = [0]; STAGES.forEach(s => CUM.push(CUM[CUM.length - 1] + s.dur));
  const TOTAL = CUM[CUM.length - 1];
  const ZOOM = 14, CEN = [640, 380];
  const lamOf = it => (1 - Math.pow(0.68, it)) / (1 - Math.pow(0.68, 6));
  const lossOf = it => 0.12 + 0.88 * Math.pow(0.55, it);   // schematic, no real values

  function stageAt(T) {
    for (let s = 0; s < STAGES.length; s++) if (T <= CUM[s + 1] + 1e-9) return [s, clamp((T - CUM[s]) / STAGES[s].dur, 0, 1)];
    return [STAGES.length - 1, 1];
  }
  function partial(P, f) { const n = Math.max(2, Math.round(f * (P.length - 1)) + 1); return P.slice(0, n); }
  function trainPath(lam) { return spherePath(REL0.map((p, i) => [lerp(p[0], REL[i][0], lam), lerp(p[1], REL[i][1], lam)])); }
  const GHOSTS = [0, 1, 2, 3, 4, 5].map(i => ptsStr(trainPath(lamOf(i))));
  const zoomT = (p, z) => {   // world transform: z = 0 identity, z = 1 fully zoomed on p
    const s = Math.pow(ZOOM, z), tx = lerp(p[0], CEN[0], z), ty = lerp(p[1], CEN[1], z);
    return `translate(${fmt(tx)},${fmt(ty)}) scale(${s.toFixed(4)}) translate(${fmt(-p[0])},${fmt(-p[1])})`;
  };
  const arcD = (th0, th1, r) => {
    const P = []; for (let i = 0; i <= 24; i++) { const a = lerp(th0, th1, i / 24); P.push([S.IC[0] + r * Math.cos(a), S.IC[1] - r * Math.sin(a)]); }
    return 'M' + P.map(p => fmt(p[0]) + ',' + fmt(p[1])).join('L');
  };

  function render(T) {
    if (!built) return;
    const [st, u] = stageAt(T);
    const L = S.L;
    const show = {};   // label -> [x, y, op]
    const put = (k, x, y, op) => { if (op > 0.005) show[k] = [x, y, op]; };

    // ---- world (sphere) ----
    let worldOp = 0, zoom = 0;
    if (st <= 1) { worldOp = st === 0 ? seg(u, 0, 0.18) : 1; }
    if (st === 2) { worldOp = 1 - seg(u, 0.05, 0.4); zoom = seg(u, 0, 0.4); }
    if (st === 5) { worldOp = seg(u, 0, 0.3); zoom = 1 - seg(u, 0, 0.32); }
    if (st === 6) worldOp = 1;
    set(S.world, { opacity: worldOp.toFixed(3), transform: zoom > 0 ? zoomT(S.pS, zoom) : '' });
    S.world.style.display = worldOp < 0.005 ? 'none' : '';   // skip drawing the sphere while it is invisible
    S.levels.setAttribute('opacity', st === 0 ? seg(u, 0.1, 0.4) : 1);
    const legOp = st === 0 ? seg(u, 0.15, 0.4) : st === 1 ? 1 : st === 6 ? 0 : 0;
    S.legend.setAttribute('opacity', legOp);
    put('legLo', 870, 114, legOp); put('legHi', 1110, 114, legOp); put('legT', 990, 82, legOp);
    put('so', SC[0] - R + 30, SC[1] - R + 20, worldOp * (st <= 1 ? 1 : st === 6 ? 1 : 0));

    // markers and the three bases
    const randOp = st === 0 ? seg(u, 0.42, 0.55) : st === 1 ? 1 - seg(u, 0, 0.2) : 0;
    const fourOp = st === 0 ? seg(u, 0.57, 0.7) : st <= 1 ? 1 : 0;
    const optOp = st === 0 ? seg(u, 0.72, 0.85) : st === 1 ? 1 : 0;
    S.mRand.setAttribute('opacity', randOp);
    S.mStart.setAttribute('opacity', st <= 1 ? fourOp : st >= 5 ? 1 : 1);
    S.mOpt.setAttribute('opacity', st <= 1 ? optOp : 1);
    S.stR.update({ x: S.pR[0] + 14, y: S.pR[1] + 12, s: 0.8, funcs: E0.map((_, k) => frameTrain(0, k)), col: C.silver, op: randOp });
    put('rand', S.pR[0] + 62, S.pR[1] + 75, randOp);
    S.stF.update({ x: S.pS[0] + 16, y: S.pS[1] - 22, s: 0.8, funcs: E0, col: C.slate, op: fourOp });
    put('four', S.pS[0] - 14, S.pS[1] + 1, st <= 1 ? fourOp : 0);
    const qX = [118, 548];
    S.stX.update({ x: qX[0], y: qX[1], s: 0.8, funcs: E0.map((_, k) => basisAt(1, k)), col: C.red, op: optOp });
    set(S.connX, { x1: S.pX[0] - 10, y1: S.pX[1] + 4, x2: qX[0] + 74, y2: qX[1] + 22, opacity: optOp });
    put('opt', qX[0] + 50, qX[1] + 66, st === 0 ? optOp : 0);
    put('learned', qX[0] + 50, qX[1] + 66, st === 1 ? optOp : 0);

    // path
    let pathOp = 0, pathFrac = 0;
    if (st === 1) { pathFrac = seg(u, 0.12, 0.7); pathOp = 1; }
    if (st === 2) { pathFrac = 1; pathOp = 1; }
    if (st === 5) { pathFrac = 1; pathOp = 1; }
    const pp = ptsStr(partial(PATH, pathFrac));
    set(S.path, { points: pp, opacity: pathOp }); set(S.pathGlow, { points: pp, opacity: pathOp });
    put('ode', 1010, 300, st === 1 ? seg(u, 0.55, 0.75) : 0);
    put('ode2', 1010, 352, st === 1 ? seg(u, 0.7, 0.9) : 0);

    // ---- tangent space: the patch at the start of the path zooms with the world, then opens into the panel ----
    let tOp = 0, tm = 0;
    if (st === 2) { tOp = seg(u, 0, 0.12); tm = seg(u, 0.3, 0.55); }
    if (st === 3 || st === 4) { tOp = 1; tm = 1; }
    if (st === 5) { tOp = 1 - seg(u, 0.28, 0.42); tm = 1 - seg(u, 0, 0.22); }
    if (tOp > 0.005) {
      const zs = Math.pow(ZOOM, zoom), zx = lerp(S.pS[0], CEN[0], zoom), zy = lerp(S.pS[1], CEN[1], zoom);
      const scr = p => [zx + zs * (p[0] - S.pS[0]), zy + zs * (p[1] - S.pS[1])];
      const RECT = [[40, 702], [1240, 702], [1240, 18], [40, 18]];
      const Q = S.tpatch.map((p, i) => { const a = scr(p); return [lerp(a[0], RECT[i][0], tm), lerp(a[1], RECT[i][1], tm)]; });
      set(S.tplane, { points: ptsStr(Q), opacity: tOp.toFixed(3), 'stroke-opacity': (0.5 * (1 - tm) + 0.15 * tm).toFixed(3) });
      const d = scr(S.pS); set(S.tdot, { cx: fmt(d[0]), cy: fmt(d[1]), opacity: (tOp * (1 - tm)).toFixed(3) });
    } else { S.tplane.setAttribute('opacity', 0); S.tdot.setAttribute('opacity', 0); }
    put('tan', 1228, 686, tOp * seg(tm, 0.85, 1));

    // ---- zoomed panel ----
    const panelOp = st === 2 ? seg(u, 0.38, 0.55) : st === 3 ? 1 : st === 4 ? 1 - seg(u, 0, 0.2) : 0;
    S.panel.setAttribute('opacity', panelOp);
    S.rows.setAttribute('opacity', st === 2 ? seg(u, 0.55, 0.75) : 1);
    S.circ.setAttribute('opacity', st === 2 ? seg(u, 0.62, 0.8) : 1);
    put('head', 640, 46, st === 2 ? seg(u, 0.4, 0.55) : st === 3 ? 1 : 0);
    const rowsOp = (st === 2 ? seg(u, 0.55, 0.75) : 1) * panelOp;
    put('a', 98, 318, rowsOp); put('b', 98, 418, rowsOp); put('q', 425, 470, rowsOp);
    const circOp = (st === 2 ? seg(u, 0.62, 0.8) : 1) * panelOp;
    put('ptitle', S.IC[0], S.IC[1] - S.RAD - 44, circOp);
    put('pa', S.IC[0] + S.RAD + 26, S.IC[1] + 18, circOp);
    put('pb', S.IC[0] + 24, S.IC[1] - S.RAD - 18, circOp);
    put('pnote', S.IC[0], S.IC[1] + S.RAD + 46, (st === 2 ? seg(u, 0.75, 0.92) : 1) * panelOp);
    const TX = [110, 275, 440, 605], TY = 92, CW = 140, CH = 82, BY = 520;
    put('topT', 110, 78, panelOp); put('botT', 110, BY - 14, st >= 3 ? panelOp : 0);
    for (let k = 0; k < 4; k++) {
      let w = 0, active = false;      // w: progress of function k's rotation
      if (st === 3) { const loc = u * 4 - k; w = seg(loc, 0.25, 0.8); active = loc > 0 && loc < 1; }
      if (st >= 4) w = 1;
      const started = st >= 4 || (st === 3 && u * 4 - k > 0.05);
      S.top[k].update({ x: TX[k], y: TY, w: CW, h: CH, f: E0[k], col: FCOL[k], op: panelOp, sw: active ? 3 : 1.3 });
      put('ek' + k, TX[k] + CW / 2, TY + CH + 16, panelOp);
      const fNew = rot(A0, B0, E0[k], DT * w);
      const bop = started ? panelOp * (st === 3 ? seg(u * 4 - k, 0.05, 0.25) : 1) : 0;
      S.bot[k].update({ x: TX[k], y: BY, w: CW, h: CH, f: fNew, ghostF: E0[k], col: FCOL[k], op: bop });
      put('en' + k, TX[k] + CW / 2, BY + CH + 18, bop);
      const ca = A0[k], cb = B0[k], th0 = Math.atan2(cb, ca), rr = Math.hypot(ca, cb) * S.RAD, th = th0 - DT * w;
      const aop = started ? panelOp * (st === 3 ? seg(u * 4 - k, 0.05, 0.2) : 1) : 0;
      S.ghostArr[k].set(S.IC[0], S.IC[1], S.IC[0] + rr * Math.cos(th0), S.IC[1] - rr * Math.sin(th0), aop * 0.9);
      S.arr[k].set(S.IC[0], S.IC[1], S.IC[0] + rr * Math.cos(th), S.IC[1] - rr * Math.sin(th), aop);
      set(S.arcs[k], { d: arcD(th0, th0 - DT * Math.max(w, 0.001), Math.min(46 + 9 * k, rr - 8)), opacity: aop * (w > 0.01 ? 1 : 0) });
      S.links[k].set(TX[k] + CW / 2, TY + CH + 30, TX[k] + CW / 2, BY - 30, active ? panelOp * 0.9 : 0);
      if (k === 0) put('dt', S.IC[0] + 0.62 * rr * Math.cos(th0 - DT / 2), S.IC[1] - 0.62 * rr * Math.sin(th0 - DT / 2) + 2, st === 3 && w > 0.95 ? aop : st === 4 ? aop : 0);
    }
    // ---- frame and Gram matrix ----
    const frOp = st === 4 ? seg(u, 0.15, 0.32) : st === 5 ? 1 - seg(u, 0, 0.2) : 0;
    S.fr.setAttribute('opacity', frOp);
    const wv = st === 4 ? seg(u, 0.38, 0.85) : 1;
    const fAfter = E0.map(e => rot(A0, B0, e, DT * wv));
    S.stBefore.update({ x: 110, y: 430, s: 1.6, funcs: E0, col: C.slate, op: 1 });
    S.stAfter.update({ x: 470, y: 430, s: 1.6, funcs: fAfter, col: mixHex(C.slate, C.violet, wv), op: 1 });
    S.frArrow.set(350, 350, 455, 350, 1);
    put('frA', 220, 548, frOp); put('frB', 580, 548, frOp); put('frOp', 403, 322, frOp);
    S.gram.setAttribute('opacity', frOp);
    if (frOp > 0.01) for (const c of S.gcells) {
      const v = dot(fAfter[c.i], fAfter[c.j]), s = Math.abs(v) < 0.005 ? '0.00' : v.toFixed(2);
      if (c.t.textContent !== s) c.t.textContent = s;
    }
    put('gT', 990, 172, frOp); put('gB', 990, 560, st === 4 ? seg(u, 0.55, 0.75) * frOp : frOp);

    // ---- riding the path ----
    const rideOp = st === 5 ? seg(u, 0.3, 0.42) : 0;
    const sR = st === 5 ? seg(u, 0.4, 0.95) : 0;
    const iR = Math.round(sR * (PATH.length - 1)), pR = PATH[iR];
    S.stRide.update({ x: pR[0] + 14, y: pR[1] - 20, s: 0.8, funcs: E0.map((_, k) => basisAt(sR, k)), col: mixHex(C.slate, C.red, sR), op: rideOp });
    if (st === 5) {
      put('four', S.pS[0] - 14, S.pS[1] + 1, rideOp);
      put('learned', pR[0] + 66, pR[1] + 52, seg(u, 0.93, 1));
      put('so', SC[0] - R + 30, SC[1] - R + 20, worldOp);
    }

    // ---- training ----
    const trOp = st === 6 ? 1 : 0;
    const it = st === 6 ? 6 * seg(u, 0.25, 0.95) : 0, lam = lamOf(it);
    if (st === 6) {
      const P = trainPath(lam), e = P[P.length - 1];
      set(S.pathT, { points: ptsStr(P), opacity: seg(u, 0, 0.12) });
      for (let i = 0; i < 6; i++) set(S.ghosts[i], { points: GHOSTS[i], opacity: it > i + 0.02 ? 0.35 : 0 });
      S.mEnd.setAttribute('opacity', seg(u, 0, 0.12)); set(S.mEnd, { cx: fmt(e[0]), cy: fmt(e[1]) });
      S.stT.update({ x: e[0] + 14, y: e[1] + 10, s: 0.8, funcs: E0.map((_, k) => frameTrain(lam, k)), col: mixHex(C.silver, C.red, lam), op: seg(u, 0.05, 0.15) });
      put('th0', e[0] - 12, e[1] + 2, 1 - seg(u, 0.18, 0.28));
      const lpt = v => [S.LO[0] + 12 + (S.LW - 40) * v / 6, S.LO[1] - (S.LH - 20) * lossOf(v)];
      const LP = []; for (let i = 0; i <= 40; i++) LP.push(lpt(it * i / 40));
      S.lcurve.setAttribute('points', ptsStr(LP)); const ld = lpt(it); set(S.ldot, { cx: fmt(ld[0]), cy: fmt(ld[1]) });
      S.mOpt.setAttribute('opacity', 1);
    } else {
      S.pathT.setAttribute('opacity', 0); S.ghosts.forEach(g => g.setAttribute('opacity', 0)); S.mEnd.setAttribute('opacity', 0);
      S.stT.update({ op: 0 });
    }
    const plotOp = st === 6 ? seg(u, 0.08, 0.22) : 0;
    S.plot.setAttribute('opacity', plotOp);
    put('ly', S.LO[0] - 10, S.LO[1] - S.LH + 10, plotOp); put('lx', S.LO[0] + S.LW / 2, S.LO[1] + 24, plotOp);
    put('lnote', S.LO[0] + S.LW, S.LO[1] - S.LH + 12, plotOp); put('rule', S.LO[0] + S.LW / 2, S.LO[1] + 76, st === 6 ? seg(u, 0.15, 0.3) : 0);
    if (st === 6) put('so', SC[0] - R + 30, SC[1] - R + 20, 1);

    // apply labels
    for (const k in L) {
      const o = L[k];
      if (Array.isArray(o)) o.forEach((x, i) => { const v = show[k + i]; x.at(v ? v[0] : 0, v ? v[1] : 0, v ? v[2] : 0); });
      else { const v = show[k]; o.at(v ? v[0] : 0, v ? v[1] : 0, v ? v[2] : 0); }
    }
  }

  // ---------- player ----------
  function Player(root) {
    build(root);
    let T = 0, target = null, playing = false, last = 0, raf = 0;
    const listeners = [];
    const reduced = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const emit = () => { const [s, u] = stageAt(T); listeners.forEach(f => f({ stage: s, progress: u, T, playing, stages: STAGES })); };
    function frame(now) {
      const dt = Math.max(0, Math.min(0.05, (now - last) / 1000)); last = now;   // never step backwards
      T = Math.min(target, T + dt);
      render(T); emit();
      if (T >= target - 1e-6) { playing = false; raf = 0; emit(); return; }
      raf = requestAnimationFrame(frame);
    }
    function playTo(k) {
      k = clamp(k, 0, STAGES.length - 1);
      target = CUM[k + 1];
      if (T >= target) T = CUM[k];
      if (reduced) { T = target; render(T); playing = false; emit(); return; }
      playing = true; last = performance.now();
      if (!raf) raf = requestAnimationFrame(frame);
      emit();
    }
    function pause() { playing = false; if (raf) cancelAnimationFrame(raf); raf = 0; emit(); }
    function jump(k) { pause(); T = CUM[clamp(k, 0, STAGES.length - 1) + 1]; render(T); emit(); }
    const cur = () => stageAt(T)[0];
    const atStop = () => CUM.some(c => Math.abs(c - T) < 1e-6);
    const api = {
      stages: STAGES,
      on(f) { listeners.push(f); emit(); },
      play() {
        if (playing) return pause();
        const s = cur();
        if (T >= CUM[s + 1] - 1e-6 && T > 0) {
          if (s + 1 < STAGES.length) playTo(s + 1);
          else { T = 0; render(T); playTo(0); }
        } else playTo(s);
      },
      next() { const s = cur(); if (playing) { jump(s); return; } if (T >= CUM[s + 1] - 1e-6) { if (s + 1 < STAGES.length) playTo(s + 1); } else playTo(s); },
      prev() { const s = cur(); jump(Math.max(0, s - 1)); },
      go(k) { jump(k); },
      replay(k) { pause(); T = CUM[k]; render(T); playTo(k); },
      seek(t) { pause(); T = clamp(t, 0, TOTAL); render(T); emit(); },
      get T() { return T; }, get playing() { return playing; },
      CUM, TOTAL,
    };
    render(0);
    return api;
  }
  window.B4SPlayer = Player;
})();
