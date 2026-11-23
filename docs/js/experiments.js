/* Experiments section: rebuilds the paper's figures natively from figdata/*.json (values and image tiles extracted from the
 * paper's figures) using the chart kit in charts.js. */
(function () {
  'use strict';
  const { lineChart, segmented, COL, fmtNum, h } = window.Charts;
  const get = name => fetch('figdata/' + name + '.json').then(r => r.json());

  function panel(parent, title, sub) {
    const p = h('div', 'panel', parent);
    h('div', 'panel-t', p, title);
    if (sub) h('div', 'panel-s', p, sub);
    return p;
  }
  function tile(parent, src, { badge, badge2, pixel, alt } = {}) {
    const f = h('figure', 'tile', parent);
    const img = h('img', pixel ? 'px' : '', f);
    img.src = src; img.alt = alt || ''; img.loading = 'lazy'; img.decoding = 'async';
    if (badge) h('span', 'badge', f, badge);
    if (badge2) h('span', 'badge b', f, badge2);
    return f;
  }
  function slider(parent, label, values, idx, fmt, onChange) {
    const w = h('label', 'slider', parent);
    h('span', '', w, label);
    const inp = h('input', '', w);
    inp.type = 'range'; inp.min = 0; inp.max = values.length - 1; inp.step = 1; inp.value = idx;
    const out = h('output', '', w, fmt(values[idx]));
    inp.addEventListener('input', () => { out.textContent = fmt(values[+inp.value]); onChange(+inp.value); });
    return { set(i) { inp.value = i; out.textContent = fmt(values[i]); } };
  }
  const cbar = (parent, stops, lo, hi, label) => {
    const b = h('div', 'cbar', parent);
    b.style.background = 'linear-gradient(90deg,' + stops.slice().reverse().join(',') + ')';
    const l = h('div', 'cbar-l', parent);
    h('span', '', l, String(lo)); h('span', '', l, label || ''); h('span', '', l, String(hi));
  };

  /* ---------------- NTK ---------------- */
  function ntk(root, d) {
    const T = 'figures/ntk/';
    const ctl = h('div', '', root);
    const row = h('div', 'panel-row ntk-row', root);
    const left = panel(row, 'Classifier output', 'Two-moons data and the network’s logit at the selected training step.');
    const lt = h('div', '', left);
    const cb = h('div', '', left);
    const right = panel(row, 'Top 4 eigenfunctions of the NTK', 'Same operator, two solvers. λ is the eigenvalue; signs are arbitrary, so colours can flip.');
    const grid = h('div', 'tiles', right);
    grid.style.gridTemplateColumns = '64px repeat(4, minmax(0, 1fr))';
    const ecb = h('div', '', right);
    function show(step) {
      lt.innerHTML = ''; cb.innerHTML = ''; grid.innerHTML = ''; ecb.innerHTML = '';
      tile(lt, T + 'logits_' + step + '.webp', { alt: 'classifier logits at step ' + step });
      const [lo, hi] = d.colorbar.logits[step];
      cbar(cb, d.cmap_high_to_low, lo, hi, 'logit');
      h('div', '', grid);
      for (let k = 1; k <= 4; k++) h('div', 'cl', grid, String(k));
      [['grid', 'Grid solver', ''], ['ours', 'Ours', 'ours']].forEach(([m, name, cls]) => {
        h('div', 'rl ' + cls, grid, name);
        for (let k = 1; k <= 4; k++) tile(grid, T + m + '_' + step + '_' + k + '.webp', { badge2: 'λ = ' + d.lambda[m][step][k - 1], alt: name + ' eigenfunction ' + k });
      });
      const [elo, ehi] = d.colorbar.eig.ours[step];
      cbar(ecb, d.cmap_high_to_low, '−', '+', 'eigenfunction value');
    }
    segmented(ctl, 'Training step', d.steps.map(s => ({ value: s, label: String(s) })), 5000, show);
    show(5000);
  }

  /* ---------------- Koopman ---------------- */
  function koopman(root, d) {
    const T = 'figures/koopman/';
    const row = h('div', 'panel-row wide', root);
    const left = panel(row, 'Rollout of a Taylor–Green vortex', 'Apply one learned step repeatedly (top), or transport with Runge–Kutta. ‖f‖ is the L² energy.');
    const grid = h('div', 'tiles koop', left);
    grid.style.gridTemplateColumns = '40px repeat(5, minmax(0, 1fr))';
    h('div', '', grid);
    const cells = {};
    d.snap_t.forEach(t => { cells['h' + t] = h('div', 'cl', grid, 't = ' + t + ' s'); });
    [['ours', 'Ours', 'ours'], ['rk2', 'RK2', ''], ['rk4', 'RK4', '']].forEach(([m, name, cls]) => {
      h('div', 'rl ' + cls, grid, name);
      d.snap_t.forEach(t => { cells[m + t] = tile(grid, T + m + '_t' + t + '.webp', { badge2: '‖f‖ ' + d.snap_norms[m][t], alt: name + ' at t = ' + t + ' s' }); });
    });
    const right = panel(row, 'L² energy over time', 'The true flow is volume-preserving, so the energy should stay at 1.');
    const E = d.energy;
    const ser = [
      { id: 'ours', name: 'Ours (learned)', color: COL.ours, marker: 'circle', width: 2.6 },
      { id: 'rk1', name: 'RK1', color: COL.rk1, marker: 'square', dash: '5 4' },
      { id: 'rk2', name: 'RK2', color: COL.rk2, marker: 'triangle', dash: '5 4' },
      { id: 'rk4', name: 'RK4', color: COL.rk4, marker: 'diamond', dash: '5 4' },
    ].map(s => Object.assign(s, { pts: E.t.map((t, i) => [t, E[s.id][i]]) }));
    let ti = 2;
    const chart = lineChart(h('div', '', right), {
      x: { type: 'lin', domain: [0, 4], label: 'time (s)', ticks: [0, 1, 2, 3, 4], name: 't =' },
      y: { type: 'lin', domain: [0.55, 1.35], label: 'L² energy', ticks: [0.6, 0.8, 1.0, 1.2] },
      series: ser, refY: 1, height: 330, xFmt: v => v.toFixed(1) + ' s', yFmt: v => v.toFixed(3),
      onHover: x => { if (x !== null && Number.isInteger(x)) { ti = x; sl.set(ti); draw(); } },
      sticky: () => ti,
    });
    function draw() {
      Object.entries(cells).forEach(([k, e]) => e.classList.toggle('sel', +k.replace(/\D+/, '') === ti));
      chart.setCursor(ti);
    }
    const ctl = h('div', '', left);
    left.insertBefore(ctl, grid);
    const sl = slider(ctl, 'Time', d.snap_t, ti, v => 't = ' + v + ' s', i => { ti = i; draw(); });
    draw();
  }

  /* ---------------- 2-D PCA: MNIST INRs and CelebA ---------------- */
  function fpca2d(root, d) {
    const T = 'figures/fpca2d/';
    const NAMES = { mnist: 'MNIST INRs', celeba: 'CelebA' };
    let ds = 'mnist';
    const ctl = h('div', '', root);
    // bases
    const pb = panel(root, 'The first 12 basis functions', 'Top: the Fourier basis we start from. Bottom: the learned basis after training.');
    pb.classList.add('panel-wide');
    const bgrid = h('div', 'tiles', pb);
    bgrid.style.gridTemplateColumns = '64px repeat(12, minmax(0, 1fr))';
    function bases() {
      bgrid.innerHTML = '';
      h('div', '', bgrid);
      for (let k = 1; k <= 12; k++) h('div', 'cl', bgrid, String(k));
      [['fourier', 'Fourier', ''], ['learned', 'Learned', 'ours']].forEach(([m, name, cls]) => {
        h('div', 'rl ' + cls, bgrid, name);
        for (let k = 1; k <= 12; k++) tile(bgrid, T + ds + '_' + m + '_' + k + '.webp', { alt: NAMES[ds] + ' ' + name + ' basis function ' + k });
      });
    }
    // reconstructions (MNIST INRs) + captured energy
    const row = h('div', 'panel-row', root);
    const pr = panel(row, 'Reconstructions (MNIST INRs)', 'Keep the first N basis functions. Numbers are PSNR in dB (higher is better).');
    const RN = d.recon.N;
    let ni = 1;
    const sl = slider(pr, 'Basis functions', RN, ni, v => 'N = ' + v, i => { ni = i; recon(); });
    const rgrid = h('div', 'tiles', pr);
    rgrid.style.gridTemplateColumns = '64px repeat(4, minmax(0, 1fr))';
    function recon() {
      const N = RN[ni];
      rgrid.innerHTML = '';
      [['fourier', 'Fourier', ''], ['ours', 'Ours', 'ours']].forEach(([m, name, cls]) => {
        h('div', 'rl ' + cls, rgrid, name);
        for (let j = 1; j <= 4; j++) {
          const p = d.recon.psnr[m][N][j - 1], other = d.recon.psnr[m === 'ours' ? 'fourier' : 'ours'][N][j - 1];
          tile(rgrid, T + 'recon_' + m + '_' + N + '_' + j + '.webp', { badge: p.toFixed(1) + ' dB' + (p > other ? ' ▲' : ''), alt: name + ' reconstruction with ' + N + ' functions' });
        }
      });
      h('div', 'rl', rgrid, 'Original');
      for (let j = 1; j <= 4; j++) tile(rgrid, T + 'original_' + j + '.webp', { alt: 'original digit' });
      if (chart && ds === 'mnist') chart.setCursor(N);
    }
    const pe = panel(row, 'Cumulative captured energy', 'Fraction of the data’s energy explained by the first N functions; mean ± one std over the dataset.');
    const cwrap = h('div', '', pe);
    let chart = null;
    function energy() {
      cwrap.innerHTML = '';
      const E = d.energy[ds];
      const N = E.N.map(n => Math.round(n));
      const mk = (m, name, color) => ({ id: m, name, color, width: 2.4,
        pts: N.map((n, i) => [n, E[m].mean[i]]), band: N.map((n, i) => [n, E[m].lo[i], E[m].hi[i]]) });
      const snap = [...new Set(N)].filter(n => n >= 1);
      chart = lineChart(cwrap, {
        x: { type: 'log', domain: [1, Math.max(...N)], label: 'number of basis functions N', name: 'N =' },
        y: { type: 'lin', domain: [0, 1], label: 'captured energy', ticks: [0, 0.25, 0.5, 0.75, 1], fmt: v => (v * 100) + '%' },
        series: [mk('ours', 'Learned (ours)', COL.ours), mk('fourier', 'Fourier', COL.fourier)],
        snapX: snap, height: 320, xFmt: v => String(v), yFmt: v => (v * 100).toFixed(1) + '%',
        onHover: x => { if (ds === 'mnist' && x !== null && RN.includes(x)) { ni = RN.indexOf(x); sl.set(ni); recon(); } },
        sticky: () => (ds === 'mnist' ? RN[ni] : null),
      });
      if (ds === 'mnist') chart.setCursor(RN[ni]);
    }
    segmented(ctl, 'Dataset', [{ value: 'mnist', label: 'MNIST (SIREN INRs)' }, { value: 'celeba', label: 'CelebA' }], ds, v => { ds = v; bases(); energy(); });
    bases(); energy(); recon();
  }

  /* ---------------- 1-D functional PCA ---------------- */
  function fpca1d(root, d) {
    const row = h('div', 'panel-row', root);
    // reconstruction of one signal
    const pr = panel(row, 'Reconstructing one signal', 'Dashed: the true signal. Keep every basis function up to the chosen maximum frequency.');
    const rctl = h('div', '', pr);
    const rwrap = h('div', '', pr);
    let cut = '16';
    function recon() {
      rwrap.innerHTML = '';
      const R = d.recon;
      lineChart(rwrap, {
        x: { type: 'lin', domain: [0, 1], label: 'x', ticks: [0, 0.25, 0.5, 0.75, 1], name: 'x =' },
        y: { type: 'lin', domain: [-1.6, 2.1], ticks: [-1, 0, 1, 2], label: 'f(x)' },
        series: [
          { id: 'truth', name: 'True signal', color: COL.truth, pts: R.truth, dash: '4 4', width: 1.6 },
          { id: 'ours', name: 'Learned (ours)', color: COL.ours, pts: R.ours[cut], width: 2.2 },
          { id: 'fourier', name: 'Fourier', color: COL.fourier, pts: R.fourier[cut], width: 2.2 },
          { id: 'pca', name: 'Grid PCA', color: COL.pca, pts: R.pca[cut], width: 1.8, dots: R.pca[cut] },
        ],
        snapX: [], height: 300, xFmt: v => v.toFixed(3), yFmt: v => v.toFixed(2),
      });
    }
    segmented(rctl, 'Max frequency', d.recon.cutoffs.map(c => ({ value: String(c), label: String(c) })), cut, v => { cut = v; recon(); });
    recon();
    // error curve
    const pe = panel(row, 'Reconstruction error vs. number of basis functions', 'Mean squared error over the distribution, log–log. Grid PCA stalls at its Nyquist limit.');
    const E = d.error;
    lineChart(h('div', '', pe), {
      x: { type: 'log', domain: [1, 128], label: 'number of basis functions N', ticks: [1, 10, 100], name: 'N =' },
      y: { type: 'log', domain: [1e-4, 1], label: 'reconstruction error' },
      series: [
        { id: 'ours', name: 'Learned (ours)', color: COL.ours, marker: 'square', pts: E.N.map((n, i) => [n, E.ours[i]]), width: 2.2 },
        { id: 'fourier', name: 'Fourier', color: COL.fourier, marker: 'circle', pts: E.N.map((n, i) => [n, E.fourier[i]]), width: 2.2 },
        { id: 'pca', name: 'Grid PCA', color: COL.pca, marker: 'triangle', pts: E.N.map((n, i) => [n, E.pca[i]]), width: 2.2 },
      ],
      height: 300, xFmt: v => String(v), yFmt: v => fmtNum(v),
    });
    // data + bases
    const row2 = h('div', 'panel-row', root);
    const ps = panel(row2, 'The data', 'Random sines with a jump at x = 0.5 (random sign and frequency). Darker curves are more frequent.');
    const maxc = Math.max(...d.samples.map(s => s.count));
    lineChart(h('div', '', ps), {
      x: { type: 'lin', domain: [0, 1], label: 'x', ticks: [0, 0.25, 0.5, 0.75, 1] },
      y: { type: 'lin', domain: [-2.1, 2.1], ticks: [-2, -1, 0, 1, 2], label: 'f(x)' },
      series: d.samples.map((s, i) => ({ id: 's' + i, name: '', noLegend: true, color: '#5d646b', opacity: 0.18 + 0.72 * s.count / maxc, width: 1.5, pts: s.pts })),
      tooltip: false, height: 260,
    });
    const pb = panel(row2, 'The first 8 basis functions', 'Fourier (top), learned (middle) and grid PCA (bottom). The learned functions put a jump at x = 0.5.');
    const sp = h('div', 'sparks', pb);
    const NSs = 'http://www.w3.org/2000/svg';
    [['fourier', 'Fourier', COL.fourier, ''], ['ours', 'Learned', COL.ours, 'ours'], ['pca', 'Grid PCA', COL.pca, '']].forEach(([m, name, col, cls]) => {
      h('div', 'rl ' + cls, sp, name);
      d.bases[m].forEach((pts, k) => {
        const s = document.createElementNS(NSs, 'svg');
        s.setAttribute('viewBox', '0 0 100 70');
        s.setAttribute('role', 'img'); s.setAttribute('aria-label', name + ' basis function ' + (k + 1));
        const mid = document.createElementNS(NSs, 'line');
        Object.entries({ x1: 50, x2: 50, y1: 4, y2: 66, stroke: '#dfe3e7', 'stroke-dasharray': '2 2' }).forEach(([a, v]) => mid.setAttribute(a, v));
        s.appendChild(mid);
        const p = document.createElementNS(NSs, 'path');
        let dd = '', pen = false;
        pts.forEach(q => { if (!q || q[1] === null) { pen = false; return; } dd += (pen ? 'L' : 'M') + (4 + q[0] * 92).toFixed(1) + ',' + (35 - q[1] * 28).toFixed(1); pen = true; });
        Object.entries({ d: dd, fill: 'none', stroke: col, 'stroke-width': 1.6, 'stroke-linejoin': 'round' }).forEach(([a, v]) => p.setAttribute(a, v));
        s.appendChild(p);
        if (m === 'pca') pts.forEach(q => { const c = document.createElementNS(NSs, 'circle'); c.setAttribute('cx', (4 + q[0] * 92).toFixed(1)); c.setAttribute('cy', (35 - q[1] * 28).toFixed(1)); c.setAttribute('r', 1.8); c.setAttribute('fill', col); s.appendChild(c); });
        sp.appendChild(s);
      });
    });
    const cl = h('div', 'sparks', pb);
    h('div', '', cl);
    for (let k = 1; k <= 8; k++) h('div', 'cl', cl, String(k));
  }

  const BUILD = { ntk, koopman, fpca2d, fpca1d };
  function init() {
    document.querySelectorAll('[data-fig]').forEach(root => {
      const name = root.dataset.fig;
      if (!BUILD[name]) return;
      get(name).then(d => BUILD[name](root, d)).catch(e => { root.textContent = 'Could not load figure data (' + name + ').'; console.error(e); });
    });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
