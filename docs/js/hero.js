/* Animated Summary figure: one sphere SO(∞), three learned paths. For each leg, a stack of Fourier basis functions
 * travels along its path and turns into the learned basis, frame by frame. The frames are computed from the
 * published checkpoints (MNIST INRs, CelebA, 1-D FPCA). The transported basis is flow(W φ_k), with W the learned
 * orthogonal synthesis over a block of Fourier atoms. The first n_synth frames follow the rotation Q(s) = exp(s L),
 * L = log W skew (dQ/ds = L Q, Q(0) = I, Q(1) = W) from the raw Fourier atoms to W φ_k; the remaining n_flow frames
 * follow the learned flow, one frame after each Cayley step. Between neighbouring frames the cards cross-fade. Geometry and frames come from
 * figures/hero/hero.json; the sphere and paths are figures/hero/apps.svg. */
(function () {
  'use strict';
  const NS = 'http://www.w3.org/2000/svg';
  const root = document.querySelector('.apps');
  if (!root) return;
  const BASE = 'figures/hero/';
  const el = (tag, attrs, parent) => {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  };
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  const ease = x => { x = clamp(x, 0, 1); return x * x * (3 - 2 * x); };
  const seg = (t, a, b) => clamp((t - a) / (b - a), 0, 1);

  Promise.all([fetch(BASE + 'apps.svg').then(r => r.text()), fetch(BASE + 'hero.json').then(r => r.json())]).then(([svgText, G]) => {
    const holder = root.querySelector('.apps-svg');
    holder.innerHTML = svgText;
    const svg = holder.querySelector('svg');
    svg.setAttribute('aria-hidden', 'true');
    const defs = svg.querySelector('defs') || el('defs', {}, svg);
    const U = G.U, V = G.V, OFF = G.OFF;

    // ---------- a stack of 4 cards; anchor = bottom-left corner of the front card ----------
    // card drawn in front, per leg (default: the first card). MNIST: the constant atom, which the flow turns into a clear "0"
    const FRONT = { mnist: 3 };
    function slotsOf(leg) {
      const f = FRONT[leg.id];
      if (f === undefined) return [0, 1, 2, 3];
      const order = [f].concat([0, 1, 2, 3].filter(k => k !== f));      // front to back
      const slots = []; order.forEach((k, i) => { slots[k] = i; });
      return slots;
    }
    function Stack(leg, edge, frame0) {
      const g = el('g', {}, svg);
      const n = 4, cards = [];
      for (let i = 0; i < 3; i++) {
        const t = n + 0.35 + 0.55 * i;
        el('circle', { cx: t * OFF[0] + 0.5 * U[0] + 0.5 * V[0], cy: t * OFF[1] + 0.5 * U[1] + 0.5 * V[1], r: 7, fill: edge }, g);
      }
      const q = [[0, 0], [U[0], U[1]], [U[0] + V[0], U[1] + V[1]], [V[0], V[1]]];
      for (let k = 0; k < n; k++) {
        const card = el('g', {}, g);
        // soft shadow from two translucent offsets (no blur filter: cheap to redraw while the stack moves)
        el('polygon', { points: q.map(p => (p[0] + 14) + ',' + (p[1] + 16)).join(' '), fill: '#1c1c1c', 'fill-opacity': 0.08 }, card);
        el('polygon', { points: q.map(p => (p[0] + 7) + ',' + (p[1] + 8)).join(' '), fill: '#1c1c1c', 'fill-opacity': 0.12 }, card);
        // unit square -> card parallelogram (image top-left at V)
        const cg = el('g', { transform: `matrix(${U[0]},${U[1]},${-V[0]},${-V[1]},${V[0]},${V[1]})` }, card);
        let set;
        if (leg.kind === 'curve') {
          el('rect', { x: 0, y: 0, width: 1, height: 1, fill: '#ffffff' }, cg);
          el('line', { x1: 0.05, x2: 0.95, y1: 0.5, y2: 0.5, stroke: '#dfe3e7', 'stroke-width': 1, 'vector-effect': 'non-scaling-stroke' }, cg);
          const pl = el('polyline', { fill: 'none', stroke: leg.color, 'stroke-width': 2.2, 'stroke-linejoin': 'round', 'vector-effect': 'non-scaling-stroke' }, cg);
          const C = leg.curves[k];
          set = phi => {
            const a = Math.floor(phi), b = Math.min(a + 1, C.length - 1), w = phi - a;
            const A = C[a], B = C[b], m = A.length - 1;
            pl.setAttribute('points', A.map((v, i) => (0.06 + 0.88 * i / m).toFixed(3) + ',' + (0.5 - 0.36 * (v + (B[i] - v) * w)).toFixed(3)).join(' '));
          };
        } else {
          const T = leg.tile;
          const box = el('svg', { x: 0, y: 0, width: 1, height: 1, viewBox: `0 0 ${T} ${T}`, preserveAspectRatio: 'none', overflow: 'hidden' }, cg);
          const src = BASE + leg.id + '-card' + (k + 1) + '.webp';
          const ia = el('image', { href: src, width: T * leg.n_frames, height: T, x: 0, y: 0, preserveAspectRatio: 'none' }, box);
          const ib = el('image', { href: src, width: T * leg.n_frames, height: T, x: 0, y: 0, preserveAspectRatio: 'none', opacity: 0 }, box);
          set = phi => {   // cross-fade between neighbouring frames
            const a = Math.floor(phi), w = phi - a;
            ia.setAttribute('x', -a * T);
            ib.setAttribute('x', -Math.min(a + 1, leg.n_frames - 1) * T);
            ib.setAttribute('opacity', w.toFixed(3));
          };
        }
        el('polygon', { points: q.map(p => p.join(',')).join(' '), fill: 'none', stroke: edge, 'stroke-width': 3, 'stroke-linejoin': 'round', 'vector-effect': 'non-scaling-stroke' }, card);
        cards[k] = { set, card };
      }
      const slots = slotsOf(leg);
      cards.forEach((c, k) => c.card.setAttribute('transform', `translate(${slots[k] * OFF[0]},${slots[k] * OFF[1]})`));
      slots.map((sl, k) => [sl, k]).sort((x, y) => y[0] - x[0]).forEach(([, k]) => g.appendChild(cards[k].card));   // back to front
      let cur = -1;
      const api = {
        g,
        frame(phi) { phi = Math.round(phi * 100) / 100; if (phi !== cur) { cur = phi; cards.forEach(c => c.set(phi)); } },
        place(x, y, s, op) { g.setAttribute('transform', `translate(${x.toFixed(1)},${y.toFixed(1)}) scale(${s})`); g.setAttribute('opacity', op.toFixed(3)); g.style.display = op < 0.005 ? 'none' : ''; },
      };
      api.frame(frame0);
      return api;
    }

    // ---------- arc-length parameterised path ----------
    function Path(leg) {
      const P = [], back = [];
      leg.path.forEach(sg => sg.pts.forEach(p => { P.push(p); back.push(sg.back); }));
      const L = [0];
      for (let i = 1; i < P.length; i++) L.push(L[i - 1] + Math.hypot(P[i][0] - P[i - 1][0], P[i][1] - P[i - 1][1]));
      const tot = L[L.length - 1];
      return {
        at(u) {
          const d = clamp(u, 0, 1) * tot;
          let i = 1; while (i < L.length - 1 && L[i] < d) i++;
          const a = P[i - 1], b = P[i], w = (d - L[i - 1]) / Math.max(L[i] - L[i - 1], 1e-9);
          return { x: a[0] + (b[0] - a[0]) * w, y: a[1] + (b[1] - a[1]) * w, back: back[i], i };
        },
        upto(u) { const h = this.at(u); return P.slice(0, h.i).concat([[h.x, h.y]]).map(p => p[0].toFixed(1) + ',' + p[1].toFixed(1)).join(' '); },
      };
    }

    const legs = G.legs.map(leg => {
      const path = Path(leg);
      const trail = el('polyline', { fill: 'none', stroke: leg.color, 'stroke-width': 20, 'stroke-linecap': 'round', 'stroke-linejoin': 'round', opacity: 0 }, svg);
      return { leg, path, trail,
               top: Stack(leg, '#3f454b', 0),
               end: Stack(leg, leg.color, leg.n_frames - 1),
               trav: Stack(leg, leg.color, 0) };
    });
    const head = el('circle', { r: 13, fill: '#ffffff', 'stroke-width': 6, opacity: 0 }, svg);
    // HTML labels
    const startLabel = root.querySelector('[data-role="start-label"]');
    const status = root.parentElement.querySelector('.apps-status');
    const endLabel = { fpca1d: root.querySelector('[data-role="end-fpca1d"]') };

    // ---------- timeline ----------
    const SWAP = 0.8, APPEAR = 0.6, TRAVEL = 8.0, ARRIVE = 0.9, HOLD = 0.8;
    const LEG_T = SWAP + APPEAR + TRAVEL + ARRIVE + HOLD;
    const END_HOLD = 2.6, RESET = 0.9;
    const TOTAL = legs.length * LEG_T + END_HOLD + RESET;
    const FP = G.fourier_place, S0 = G.start;
    const DIM = 0.28, SYN = 0.5;

    function render(t) {
      t = ((t % TOTAL) + TOTAL) % TOTAL || 0;
      const li = Math.min(Math.floor(t / LEG_T), legs.length);       // legs.length = end phase
      const lt = t - li * LEG_T;
      legs.forEach((L, j) => {
        const lg = L.leg;
        // end stacks: dim until their leg arrives; fade back at the reset
        let endOp = DIM, pulse = 1;
        if (j < li) endOp = 1;
        if (j === li) { const a = seg(lt, SWAP + APPEAR + TRAVEL, SWAP + APPEAR + TRAVEL + ARRIVE); endOp = DIM + (1 - DIM) * ease(a); pulse = 1 + 0.06 * Math.sin(Math.PI * a); }
        if (li === legs.length) { const r = seg(lt, END_HOLD, END_HOLD + RESET); endOp = 1 - (1 - DIM) * ease(r); }
        const cx = lg.place[0] + U[0] / 2, cy = lg.place[1];
        L.end.place(cx - (U[0] / 2) * pulse, cy, pulse, endOp);
        // the Fourier stack at the start shows the current leg's starting basis
        let topOp = 0;
        if (j === li) topOp = ease(seg(lt, 0, SWAP));
        if (j === li - 1 && li < legs.length) topOp = 1 - ease(seg(lt, 0, SWAP));
        if (li === legs.length && j === legs.length - 1) topOp = 1 - ease(seg(lt, END_HOLD, END_HOLD + RESET));
        if (li === legs.length && j === 0) topOp = ease(seg(lt, END_HOLD, END_HOLD + RESET));
        L.top.place(FP[0], FP[1], 1, topOp);
        // traveller
        let tOp = 0, u = 0;
        if (j === li) {
          const a = seg(lt, SWAP, SWAP + APPEAR), b = seg(lt, SWAP + APPEAR, SWAP + APPEAR + TRAVEL), c = seg(lt, SWAP + APPEAR + TRAVEL, SWAP + APPEAR + TRAVEL + ARRIVE);
          u = ease(b);
          tOp = ease(a) * (1 - ease(c));
          const h = L.path.at(u);
          if (h.back) tOp *= 0.45;
          // frame position: the rotation exp(sL) over the first SYN of the trip, then one Cayley step after another
          const ns = lg.n_synth;
          // s moves slowly at first (the cards change fastest near s = 0), then speeds up: s = r^1.7
          const phi = u < SYN ? Math.pow(u / SYN, 1.7) * (ns - 1) : (ns - 1) + (u - SYN) / (1 - SYN) * lg.n_flow;
          L.trav.frame(phi);
          L.trav.place(h.x + 18, h.y - 18, 0.62, tOp);
          L.trail.setAttribute('points', L.path.upto(u));
          L.trail.setAttribute('opacity', (0.22 * ease(a) * (1 - ease(seg(lt, SWAP + APPEAR + TRAVEL + ARRIVE, LEG_T)))).toFixed(3));
          head.setAttribute('cx', h.x.toFixed(1)); head.setAttribute('cy', h.y.toFixed(1));
          head.setAttribute('stroke', lg.color);
          head.setAttribute('opacity', (ease(a) * (1 - ease(c)) * (h.back ? 0.45 : 1)).toFixed(3));
          if (status) {
            const stage = 'step ' + Math.round(phi) + ' of ' + (lg.n_frames - 1);
            status.textContent = (tOp > 0.01 ? lg.end_label + ' · ' + stage : '');
          }
        } else {
          L.trav.place(0, 0, 0.62, 0);
          L.trail.setAttribute('opacity', 0);
        }
      });
      if (li === legs.length) { head.setAttribute('opacity', 0); if (status) status.textContent = ''; }
      const cur = legs[Math.min(li, legs.length - 1)].leg;
      if (startLabel) startLabel.textContent = li === legs.length && t - li * LEG_T > END_HOLD ? legs[0].leg.start_label : cur.start_label;
    }

    // ---------- player: pause button, reduced motion, pause offscreen ----------
    const btn = root.parentElement.querySelector('.apps-play');
    const reduced = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    let t = 0, last = 0, raf = 0, playing = !reduced, visible = true;
    function tick(now) {
      // the first frame's timestamp can be earlier than performance.now() at the request: never step backwards
      const dt = clamp((now - last) / 1000, 0, 0.05); last = now;
      t = (t + dt) % TOTAL;
      render(t);
      showPos();
      raf = requestAnimationFrame(tick);
    }
    function run() {
      cancelAnimationFrame(raf); raf = 0;
      if (playing && visible) { last = performance.now(); raf = requestAnimationFrame(tick); }
      if (btn) { btn.dataset.state = playing ? 'pause' : 'play'; btn.setAttribute('aria-label', playing ? 'Pause animation' : 'Play animation'); }
    }
    if (btn) btn.addEventListener('click', () => { playing = !playing; run(); });
    // scrubber: follows the animation; dragging pauses it and jumps to that moment
    const scrub = root.parentElement.querySelector('.apps-scrub input');
    const ticks = root.parentElement.querySelector('.apps-ticks');
    if (ticks) legs.forEach((L, j) => {
      const m = document.createElement('span');
      m.style.left = (100 * (j * LEG_T + SWAP) / TOTAL) + '%';
      m.style.background = L.leg.color;
      ticks.appendChild(m);
    });
    const showPos = () => { if (scrub) scrub.value = Math.round(1000 * t / TOTAL); };
    if (scrub) scrub.addEventListener('input', () => {
      if (playing) { playing = false; run(); }
      t = +scrub.value / 1000 * TOTAL;
      render(t);
    });
    if ('IntersectionObserver' in window) new IntersectionObserver(es => { visible = es[0].isIntersecting; run(); }).observe(root);
    // reduced motion: the finished state (all three learned bases), no movement
    const fixed = /hero-t=([\d.]+)/.exec(location.hash);   // testing: #hero-t=<seconds> freezes the animation
    if (fixed) { playing = false; t = +fixed[1]; render(t); showPos(); if (btn) btn.dataset.state = 'play'; return; }
    t = reduced ? legs.length * LEG_T + 0.1 : 0;
    render(t); showPos();
    run();
  }).catch(e => console.error('summary figure', e));
})();
