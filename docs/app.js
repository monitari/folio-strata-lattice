'use strict';

// ---------------------------------------------------------------- 상수
// 색 = 모델 (후보는 범주 색, 기준 모델은 무채색). 대조군은 원래 모델의 색에 빈 원.
const MODELS = {
  transformer: { name: 'Transformer', color: 'var(--tf)' },
  gru: { name: 'GRU', color: 'var(--gru)' },
  folio: { name: 'Folio', color: 'var(--folio)' },
  strata: { name: 'Strata', color: 'var(--strata)' },
  lattice: { name: 'Lattice', color: 'var(--lattice)' },
  folio_nomem: { name: '기억 제거', base: 'folio' },
  folio_r1: { name: '읽기 1회', base: 'folio' },
  strata_sameperiod: { name: '모든 층 매번 갱신', base: 'strata' },
  strata_noaux: { name: '보조 손실 제거', base: 'strata' },
  lattice_k1: { name: '반복 1회', base: 'lattice' },
  lattice_nonbr: { name: '이웃 메시지 제거', base: 'lattice' },
  folio2: { name: 'Folio2', color: 'var(--folio2)', v1: 'folio' },
  lattice2: { name: 'Lattice2', color: 'var(--lattice2)', v1: 'lattice' },
  transformer256: { name: 'Transformer 256', color: 'var(--tf256)' },
  rule: { name: '단순 규칙', color: 'var(--muted)' },
};
const MAIN = ['transformer', 'gru', 'folio', 'strata', 'lattice'];
const CANDS = ['folio', 'strata', 'lattice'];
const IMPROVED = ['folio2', 'lattice2'];
const SUITE_ROSTER = ['transformer', 'transformer256', 'gru', 'folio', 'folio2', 'strata', 'lattice', 'lattice2'];
const COND = {
  A: { label: '공동 · 지속 기억 · 800걸음', f: (r) => r.mode === 'joint' && r.memory === 'persistent' && r.steps === 800 },
  B: { label: '공동 · 창 초기화 · 800걸음', f: (r) => r.mode === 'joint' && r.memory === 'window' && r.steps === 800 },
  D: { label: '과제별 · 지속 기억 · 400걸음', f: (r) => r.mode === 'separate' && r.memory === 'persistent' && r.steps === 400 },
  E: { label: '공동 · 지속 기억 · 3000걸음', f: (r) => r.mode === 'joint' && r.memory === 'persistent' && r.steps === 3000 },
};
const DESC = {
  switch_fast: '요리사 4명이 각자 습관대로 요리하며 평균 45접시마다 몰래 교대합니다. 지금 누가 요리하는지 알아채야 합니다.',
  phrase: '4–8접시짜리 코스 6가지가 무작위로 이어집니다. 코스가 시작되면 나머지를 이어 맞힙니다.',
  cycle: '정해진 24접시 순서가 계속 반복됩니다. 24개를 통째로 기억해야 합니다.',
  kv: '"키 3에 값 5를 저장" 같은 지시가 이어지고, 가끔 "키 3의 값은?"을 묻습니다. 다시 저장하면 덮어씁니다.',
  mixed: '빠르게 도는 짧은 무늬에, 수십 기호마다 바뀌는 느린 값이 더해진 신호입니다.',
  chain: '뒤섞인 사실 "가 → 나" 들을 보여 준 뒤 "가에서 끝까지 따라가면?"을 묻습니다. 사슬 2개, 1–3단계.',
  kv_long: '저장하고 120–300기호 뒤에 묻습니다. 그 사이에는 다른 키만 씁니다.',
  chain_long: '4–7단계 사슬 하나. 끝 개체만 찾으면 다 풀려서 해석에서 뺍니다.',
  mixed_long: '혼합 신호를 학습 때(256)보다 4배 긴 1024기호로.',
  chain_hop3: '사슬 둘(3단계 + 3단계), 두 머리를 모두 묻습니다. 학습 범위의 가장 긴 사슬.',
  chain_hop4: '사슬 둘(4단계 + 2단계), 4단계 질의만 채점. 학습보다 1단계 김.',
  chain_hop5: '사슬 둘(5단계 + 1단계), 5단계 질의만 채점. 학습보다 2단계 김.',
  kvd: '키 2개를 저장하고, 다른 키만 쓰는 방해 구간(6–192기호) 뒤에 그 둘을 묻습니다. 64기호를 넘는 지연도 학습에 들어 있습니다.',
  kvd_long: '같은 방식으로 방해 구간을 240–900기호로 늘린 시험.',
  kvd_all: '기호열은 키-값 지연 사다리와 같고, 방해 구간의 질의까지 채점하는 학습용 과제(G). 시험은 숨긴 키 질의만 봅니다.',
  cyc: '개체 8개가 하나의 고리를 이루는 사실 8개를 섞어 보여 주고, "가에서 k걸음 가면?"을 묻습니다. 고리라서 끝 개체가 없습니다.',
  count: '요리 기호가 흐르는 사이에 셀 기호(31)가 가끔 나옵니다. 가끔 "지금까지 셀 기호 개수를 m으로 나눈 나머지는?"을 묻습니다(m = 2·3·5).',
  copy: '무작위 기호 n개 뒤에 표시가 오면, 그 기호들을 그대로(29) 또는 거꾸로(30) 다시 냅니다.',
  paren: '두 종류 괄호를 겹쳐 열고 닫습니다. 닫을 때마다 어느 괄호가 닫힐지 맞힙니다. 연달아 닫는 자리만 채점합니다.',
  slowp: '빠른 무늬(3기호)에 오프셋이 더해지고, 오프셋 4개가 주기 P로 되풀이됩니다. 첫 주기 뒤에는 오프셋이 바뀌는 자리도 예측할 수 있습니다.',
  cyc_far: '같은 고리에서 4–7걸음. 학습에서는 1–3걸음만 물었습니다.',
  count_long: '같은 셈을 1024기호 동안. 학습 기록(256)의 4배입니다.',
  copy_long: '원문 길이 n = 28–48. 학습에서는 4–24였습니다.',
  paren_deep: '최대 깊이 8. 학습에서는 4까지만 쌓았습니다.',
  slowp_long: '1024기호, 주기 P = 256 포함. 학습에서는 128까지였습니다.',
};
const SVGNS = 'http://www.w3.org/2000/svg';

const $ = (s) => document.querySelector(s);
const pct = (v, d = 1) => (v == null || Number.isNaN(v) ? '–' : `${(v * 100).toFixed(d)}%`);
const p1 = (v) => (v == null || Number.isNaN(v) ? '–' : (v * 100).toFixed(1));
const num = (v) => Number(v).toLocaleString('ko-KR');
const mean = (a) => (a.length ? a.reduce((s, v) => s + v, 0) / a.length : NaN);
const sd = (a) => { if (a.length < 2) return 0; const m = mean(a); return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / (a.length - 1)); };
const nameOf = (k) => (MODELS[k].base ? `${MODELS[MODELS[k].base].name} ${MODELS[k].name}` : MODELS[k].name);
const colorOf = (k) => MODELS[MODELS[k].base || k].color;
const bytes = (b) => (b >= 2 ** 20 ? `${(b / 2 ** 20).toFixed(1)} MiB` : b >= 1024 ? `${(b / 1024).toFixed(1)} KiB` : `${num(b)} B`);

let D = null, TASK = {};
let cond = 'A', metric = 'accuracy', ablTask = 'chain', delayTrain = 'kvd';

// ---------------------------------------------------------------- 자료 꺼내기
function runsOf(variant, f) { return D.runs.filter((r) => r.variant === variant && f(r)); }
function valueOf(r, t, m = 'accuracy') {
  if (TASK[t] && TASK[t].posthoc) return r.posthoc ? r.posthoc[t] : undefined;
  return r.test[t] ? r.test[t][m] : undefined;
}
function vals(variant, f, t, m = 'accuracy') {
  return runsOf(variant, f).map((r) => valueOf(r, t, m)).filter((v) => v != null);
}
function oracleOf(t) {
  if (TASK[t].posthoc) return 1;
  const r = D.runs.find((x) => x.test[t]);
  return r ? r.test[t].oracle : null;
}

// ---------------------------------------------------------------- SVG 도우미
function el(tag, attrs = {}, parent) {
  const n = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (parent) parent.appendChild(n);
  return n;
}
function text(parent, x, y, str, attrs = {}) {
  const t = el('text', { x, y, ...attrs }, parent);
  t.textContent = str;
  return t;
}
function makeSvg(container, height, minWidth = 260) {
  container.innerHTML = '';
  const width = Math.max(minWidth, container.clientWidth);
  const svg = el('svg', { viewBox: `0 0 ${width} ${height}`, height }, container);
  return { svg, width };
}
function niceStep(range, target) {
  const raw = range / target, mag = 10 ** Math.floor(Math.log10(raw));
  return [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) || raw;
}
function swatch(key, kind = '') {
  const s = document.createElement('span');
  s.className = `sw ${kind}`;
  s.style.background = colorOf(key);
  s.style.borderColor = colorOf(key);
  return s;
}
function keyRow(container, items) {
  container.innerHTML = '';
  for (const [key, kind, label] of items) {
    const sp = document.createElement('span');
    sp.appendChild(swatch(key, kind));
    sp.append(label || nameOf(key));
    container.appendChild(sp);
  }
}
function refKey(container, items) {
  for (const [cls, label] of items) {
    const sp = document.createElement('span');
    const s = document.createElement('span');
    s.className = `sw ${cls}`;
    s.style.borderColor = 'var(--ink-2)';
    sp.append(s, label);
    container.appendChild(sp);
  }
}

// ---------------------------------------------------------------- 도움말 상자
const tip = () => $('#tip');
function showTip(ev, title, rows) {
  const t = tip();
  t.innerHTML = `<div class="tt-h">${title}</div>` + rows.map(([k, v, c]) =>
    `<div class="row">${c ? `<span class="sw" style="background:${c}"></span>` : ''}<span>${k}</span><b>${v}</b></div>`).join('');
  t.hidden = false;
  const pad = 14, w = t.offsetWidth, h = t.offsetHeight;
  let x = ev.clientX + pad, y = ev.clientY + pad;
  if (x + w > innerWidth - 8) x = ev.clientX - w - pad;
  if (y + h > innerHeight - 8) y = ev.clientY - h - pad;
  t.style.left = `${Math.max(8, x)}px`;
  t.style.top = `${Math.max(8, y)}px`;
}
function hideTip() { tip().hidden = true; }

// ---------------------------------------------------------------- 그림: 점 행 (모델 = 행, 난수 = 작은 점, 평균 = 큰 점)
function dotRows(container, { rows, domain, refs = [], fmt = p1, unit = '%', tickFmt = (v) => `${Math.round(v * 100)}` }) {
  const rowH = 24, top = 22, bottom = 24, left = 118, right = 14;
  const height = top + rows.length * rowH + bottom;
  const { svg, width } = makeSvg(container, height);
  const [lo, hi] = domain;
  const x = (v) => left + ((v - lo) / (hi - lo)) * (width - left - right);
  const step = niceStep(hi - lo, Math.max(3, Math.floor((width - left) / 70)));
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) {
    el('line', { x1: x(v), x2: x(v), y1: top - 4, y2: height - bottom, class: 'grid' }, svg);
    text(svg, x(v), height - bottom + 14, tickFmt(v), { 'text-anchor': 'middle', class: 't-num' });
  }
  // 기준선 이름표: 가운데 → 선 오른쪽 → 선 왼쪽 순으로 빈자리를 찾고, 없으면 생략 (선은 남김)
  const placed = [];
  for (const r of refs) {
    if (r.v == null || r.v < lo || r.v > hi) continue;
    const X = x(r.v);
    el('line', { x1: X, x2: X, y1: top - 4, y2: height - bottom, class: r.cls || 'ref' }, svg);
    const t = text(svg, X, top - 9, r.label, { 'text-anchor': 'middle', class: 't-ink2' });
    const w = t.getComputedTextLength() || r.label.length * 7;
    const fit = [['middle', X, X - w / 2], ['start', X + 3, X + 3], ['end', X - 3, X - 3 - w]]
      .find(([, , a]) => a >= 0 && a + w <= width && placed.every(([p, q]) => a + w + 6 < p || a > q + 6));
    if (!fit) { t.remove(); continue; }
    t.setAttribute('text-anchor', fit[0]);
    t.setAttribute('x', fit[1]);
    placed.push([fit[2], fit[2] + w]);
  }
  rows.forEach((row, i) => {
    const cy = top + i * rowH + rowH / 2;
    text(svg, left - 10, cy + 4, row.label, { 'text-anchor': 'end', class: 't-ink' });
    const c = colorOf(row.key), ring = !!MODELS[row.key].base;
    if (!row.values.length) {
      text(svg, left + 4, cy + 4, '실행 없음', { class: 't-ink2' });
      return;
    }
    const m = mean(row.values), s = sd(row.values);
    for (const v of row.values) {
      el('circle', { cx: x(v), cy, r: 3, style: `fill:${c};fill-opacity:0.35` }, svg);
    }
    if (row.values.length > 1) {
      el('line', { x1: x(Math.max(lo, m - s)), x2: x(Math.min(hi, m + s)), y1: cy, y2: cy, style: `stroke:${c};stroke-width:2` }, svg);
    }
    el('circle', { cx: x(m), cy, r: 5.5, style: ring ? `fill:var(--page);stroke:${c};stroke-width:2` : `fill:${c};stroke:var(--page);stroke-width:2` }, svg);
    const hit = el('rect', { x: 0, y: cy - rowH / 2, width, height: rowH, class: 'hit' }, svg);
    hit.addEventListener('mousemove', (ev) => showTip(ev, row.label, [
      ['평균', `${fmt(m)}${unit}`, c],
      ...(row.values.length > 1 ? [['표준편차', `${fmt(s)}${unit}`]] : []),
      ['난수별', row.values.map((v) => fmt(v)).join(' · ')],
      ...(row.extra || []),
    ]));
    hit.addEventListener('mouseleave', hideTip);
  });
}

// ---------------------------------------------------------------- 그림: 선 (x = 걸음 또는 단계)
function lineChart(container, { series, xDomain, yDomain = [0, 1], refs = [], vlines = [], xTicks, xFmt = (v) => num(v), height = 210 }) {
  const top = 12, bottom = 26, left = 34, right = 70;
  const { svg, width } = makeSvg(container, height);
  const x = (v) => left + ((v - xDomain[0]) / (xDomain[1] - xDomain[0])) * (width - left - right);
  const y = (v) => top + (1 - (v - yDomain[0]) / (yDomain[1] - yDomain[0])) * (height - top - bottom);
  for (let v = yDomain[0]; v <= yDomain[1] + 1e-9; v += 0.25) {
    el('line', { x1: left, x2: width - right, y1: y(v), y2: y(v), class: 'grid' }, svg);
    text(svg, left - 6, y(v) + 4, p1(v).replace('.0', ''), { 'text-anchor': 'end', class: 't-num' });
  }
  for (const v of xTicks) text(svg, x(v), height - 8, xFmt(v), { 'text-anchor': 'middle', class: 't-num' });
  el('line', { x1: left, x2: width - right, y1: y(yDomain[0]), y2: y(yDomain[0]), class: 'axis' }, svg);
  for (const r of refs) {
    if (r.v == null) continue;
    el('line', { x1: left, x2: width - right, y1: y(r.v), y2: y(r.v), class: r.cls || 'ref' }, svg);
  }
  for (const v of vlines) {
    el('line', { x1: x(v.x), x2: x(v.x), y1: top, y2: height - bottom, class: 'ref' }, svg);
    text(svg, x(v.x) + 4, top + 10, v.label, { class: 't-ink2' });
  }
  const ends = [];
  for (const s of series) {
    if (!s.points.length) continue;
    const c = colorOf(s.key);
    const d = s.points.map(([a, b], i) => `${i ? 'L' : 'M'}${x(a).toFixed(1)},${y(b).toFixed(1)}`).join('');
    el('path', { d, fill: 'none', style: `stroke:${c};stroke-width:${s.thin ? 1.25 : 2};stroke-opacity:${s.thin ? 0.55 : 1};stroke-linejoin:round${s.dash ? ';stroke-dasharray:4 3' : ''}` }, svg);
    if (s.dots) for (const [a, b] of s.points) el('circle', { cx: x(a), cy: y(b), r: 3.5, style: `fill:${c};stroke:var(--page);stroke-width:1.5` }, svg);
    if (s.label) { const [a, b] = s.points[s.points.length - 1]; ends.push({ y: y(b), x: x(a), label: s.label }); }
  }
  // 끝 이름표: 겹치지 않게 아래로 밀고, 넘치면 위로 당긴다
  ends.sort((a, b) => a.y - b.y);
  for (let i = 1; i < ends.length; i++) ends[i].y = Math.max(ends[i].y, ends[i - 1].y + 12);
  const maxY = height - bottom;
  for (let i = ends.length - 1; i >= 0; i--) {
    if (ends[i].y > maxY) ends[i].y = maxY;
    if (i < ends.length - 1 && ends[i].y > ends[i + 1].y - 12) ends[i].y = ends[i + 1].y - 12;
  }
  for (const e of ends) text(svg, e.x + 6, e.y + 4, e.label, { class: 't-ink2' });
  // 십자선 + 도움말: 가장 가까운 x 의 값
  const xs = [...new Set(series.flatMap((s) => s.points.map((p) => p[0])))].sort((a, b) => a - b);
  const cross = el('line', { y1: top, y2: height - bottom, class: 'axis', visibility: 'hidden' }, svg);
  const hit = el('rect', { x: left, y: top, width: width - left - right, height: height - top - bottom, class: 'hit' }, svg);
  hit.addEventListener('mousemove', (ev) => {
    const box = svg.getBoundingClientRect();
    const mx = ((ev.clientX - box.left) / box.width) * width;
    const v0 = xDomain[0] + ((mx - left) / (width - left - right)) * (xDomain[1] - xDomain[0]);
    const near = xs.reduce((a, b) => (Math.abs(b - v0) < Math.abs(a - v0) ? b : a), xs[0]);
    cross.setAttribute('x1', x(near)); cross.setAttribute('x2', x(near)); cross.setAttribute('visibility', 'visible');
    const rows = series.map((s) => { const p = s.points.find((q) => q[0] === near); return p ? [s.tipLabel || nameOf(s.key), pct(p[1]), colorOf(s.key)] : null; }).filter(Boolean);
    showTip(ev, xFmt(near), rows);
  });
  hit.addEventListener('mouseleave', () => { cross.setAttribute('visibility', 'hidden'); hideTip(); });
}

function panel(container, title, sub) {
  const div = document.createElement('div');
  div.className = 'panel';
  div.innerHTML = `<h4>${title}</h4>${sub ? `<p class="sub">${sub}</p>` : ''}`;
  const chart = document.createElement('div');
  chart.className = 'chart';
  div.appendChild(chart);
  container.appendChild(div);
  return chart;
}
function picker(container, label, options, current, onPick) {
  container.innerHTML = `<span class="lab">${label}</span>`;
  for (const [k, name] of options) {
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = name;
    b.setAttribute('aria-pressed', String(k === current));
    b.addEventListener('click', () => onPick(k));
    container.appendChild(b);
  }
}
function refsFor(t) {
  const T = TASK[t], out = [];
  const o = oracleOf(t);
  if (o != null) out.push({ v: o, label: `최대 ${p1(o).replace('.0', '')}`, cls: 'ref' });
  if (T.rule) out.push({ v: T.rule.accuracy, label: `규칙 ${p1(T.rule.accuracy).replace('.0', '')}`, cls: 'ref-rule' });
  out.push({ v: T.chance, label: '우연 12.5', cls: 'grid' });
  return out;
}

// ---------------------------------------------------------------- 1. 후보 그림
function drawSchematics() {
  const g = $('#folio-slots');
  g.innerHTML = '';
  for (let i = 0; i < 32; i++) {
    const cx = 140 + (i % 8) * 19, cy = 18 + Math.floor(i / 8) * 15;
    el('rect', { x: cx, y: cy, width: 15, height: 11, rx: 1.5, style: `fill:${i === 9 || i === 22 ? 'var(--folio)' : 'none'};stroke:var(--folio);stroke-width:1;fill-opacity:0.6` }, g);
  }
  const s = $('#strata-svg');
  s.innerHTML = '';
  // 층마다 한 번 고쳐질 때까지의 구간을 한 칸으로 그린다 (64기호 동안)
  const u = 3.8;
  [64, 16, 4, 1].forEach((p, row) => {
    const y = 8 + row * 20;
    text(s, 0, y + 10, `${p}마다`, { class: 't-num' });
    for (let t = 0; t < 64; t += p) {
      el('rect', { x: 50 + t * u, y, width: Math.max(1.6, p * u - 1.6), height: 13, rx: 1.5, style: `fill:var(--strata);fill-opacity:${0.3 + row * 0.18}` }, s);
    }
  });
  const l = $('#lattice-svg');
  l.innerHTML = '';
  const cell = 17, ox = 92;
  for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) {
    el('rect', { x: ox + c * (cell + 4), y: 4 + r * (cell + 4), width: cell, height: cell, rx: 2, style: `fill:${r === 0 && c === 0 ? 'var(--lattice)' : 'none'};stroke:var(--lattice);stroke-width:1.2` }, l);
  }
  // 가운데 칸 하나가 상하좌우 이웃과 주고받는 모습
  const cx = ox + 2 * (cell + 4) + cell / 2, cy = 4 + 2 * (cell + 4) + cell / 2;
  el('circle', { cx, cy, r: 2.5, style: 'fill:var(--ink-2)' }, l);
  for (const [dx, dy] of [[0, -1], [0, 1], [-1, 0], [1, 0]]) {
    el('line', { x1: cx + dx * 4, y1: cy + dy * 4, x2: cx + dx * (cell + 1), y2: cy + dy * (cell + 1), style: 'stroke:var(--ink-2);stroke-width:1.2', 'marker-end': 'url(#arr)' }, l);
  }
  text(l, ox - 10, 16, '입력', { 'text-anchor': 'end', class: 't-num' });
  text(l, ox + 96, 50, '× 3회 반복', { class: 't-num' });
}

// ---------------------------------------------------------------- 2. 과제 표
function drawTasks() {
  const rows = D.tasks;
  const head = '<thead><tr><th class="l">문제</th><th class="l">무엇을 보나</th><th>종류</th><th>길이</th><th>이론적 최대</th><th>단순 규칙</th><th>우연</th></tr></thead>';
  const group = (title, list) => `<tr class="group"><td colspan="7">${title}</td></tr>` + list.map((t) => `<tr>
      <td class="l"><b>${t.name}${t.key === 'chain_long' ? ' *' : ''}</b><span class="sub">${t.skill}</span></td>
      <td class="l" style="max-width:380px">${DESC[t.key] || ''}</td>
      <td>${t.kind === 'query' ? '질의' : '흐름'}</td>
      <td class="n">${num(t.length)}</td>
      <td class="n">${pct(oracleOf(t.key))}</td>
      <td class="n">${t.rule ? `${pct(t.rule.accuracy)}<span class="sub">${t.rule.name}</span>` : '<span class="dim">–</span>'}</td>
      <td class="n">${pct(t.chance)}</td></tr>`).join('');
  $('#task-table').innerHTML = head + '<tbody>' + group('학습과 시험', rows.filter((t) => t.train && t.core))
    + group('시험만 (학습보다 길거나 멀다)', rows.filter((t) => !t.train && t.core))
    + group('추가 시험 (관계 사슬 단계 수, 5절)', rows.filter((t) => t.posthoc))
    + group('추가 실험 F · G (키-값 지연 사다리, 6절)', rows.filter((t) => !t.core && !t.posthoc && !t.suite))
    + group('새 과제 묶음 H · 학습과 시험 (7절)', rows.filter((t) => t.suite && t.train))
    + group('새 과제 묶음 H · 시험만 (더 어렵게)', rows.filter((t) => t.suite && !t.train)) + '</tbody>';
}

// ---------------------------------------------------------------- 3. 정확도
function drawAccuracy() {
  picker($('#cond-picker'), '조건', Object.entries(COND).map(([k, c]) => [k, c.label]), cond, (k) => { cond = k; drawAccuracy(); });
  picker($('#metric-picker'), '지표', [['accuracy', '정확도'], ['bits', '손실 (bits, 낮을수록 좋음)']], metric, (k) => { metric = k; drawAccuracy(); });
  const key = $('#acc-key');
  const f = COND[cond].f;
  const roster = MAIN.concat(IMPROVED.filter((v) => runsOf(v, f).length));
  keyRow(key, roster.map((k) => [k, '']));
  if (metric === 'accuracy') refKey(key, [['line', '이론적 최대'], ['dash', '단순 규칙'], ['thin', '우연']]);
  const tasks = D.tasks.filter((t) => !t.posthoc && !t.suite && MAIN.some((v) => vals(v, f, t.key).length));
  const box = $('#acc-charts');
  box.innerHTML = '';
  for (const t of tasks) {
    const rows = roster.map((v) => ({ key: v, label: nameOf(v), values: vals(v, f, t.key, metric) }));
    const sub = `${t.skill}${t.train ? '' : ' · 시험만'}${t.key === 'chain_long' ? ' · 끝 개체만 찾아도 풀림' : ''}`;
    const chart = panel(box, t.name + (t.key === 'chain_long' ? ' *' : ''), sub);
    if (metric === 'accuracy') {
      dotRows(chart, { rows, domain: [0, 1], refs: refsFor(t.key) });
    } else {
      const all = rows.flatMap((r) => r.values);
      const hi = Math.max(3.2, Math.ceil(Math.max(...all) * 2) / 2);
      dotRows(chart, { rows, domain: [0, hi], refs: [{ v: 3, label: '8개 중 찍기', cls: 'grid' }], fmt: (v) => v.toFixed(2), unit: ' bits', tickFmt: (v) => `${+v.toFixed(1)}` });
    }
  }
  const n = new Set(D.runs.filter(f).map((r) => r.seed)).size;
  $('#acc-cap').innerHTML = `<b>${COND[cond].label}.</b> 난수 ${n}개. 작은 점이 난수 하나, 굵은 점이 평균, 가로 막대가 ±표준편차입니다. `
    + (cond === 'D' ? '과제별 학습은 그 문제와 그 문제의 긴 변형만 시험했습니다. ' : '')
    + (cond === 'E' && n === 1 ? '난수 하나뿐이라 편차가 없습니다. ' : '')
    + (roster.length > MAIN.length ? 'Folio2 · Lattice2 는 개선판입니다(8절). ' : '')
    + (metric === 'bits' ? '손실은 정답에 준 확률의 로그(비트)로, 0에 가까울수록 확신을 갖고 맞힌 것입니다.' : '');
}

// ---------------------------------------------------------------- 4. 학습 곡선
function meanCurve(variant, f, t) {
  // 같은 조건에 다른 과제 묶음의 실행(예: 실험 H)이 섞일 수 있으므로 이 과제를 학습한 실행만 쓴다
  const rs = runsOf(variant, f).filter((r) => r.tasks.includes(t));
  if (!rs.length) return [];
  return rs[0].history.map((h, i) => [h.step, mean(rs.map((r) => (r.history[i] ? r.history[i].val[t] : NaN)).filter((v) => !Number.isNaN(v)))]);
}
function drawCurves() {
  const key = $('#curve-key');
  keyRow(key, MAIN.map((k) => [k, 'line']));
  refKey(key, [['thin', '800걸음 (평균)'], ['line', '이론적 최대'], ['dash', '단순 규칙']]);
  const box = $('#curve-charts');
  box.innerHTML = '';
  for (const t of D.tasks.filter((x) => x.train && x.core)) {
    const series = [];
    for (const v of MAIN) {
      series.push({ key: v, points: meanCurve(v, COND.A.f, t.key), thin: true, tipLabel: `${MODELS[v].name} 800` });
      series.push({ key: v, points: meanCurve(v, COND.E.f, t.key), label: MODELS[v].name, tipLabel: `${MODELS[v].name} 3000` });
    }
    const refs = [{ v: oracleOf(t.key), cls: 'ref' }];
    if (t.rule) refs.push({ v: t.rule.accuracy, cls: 'ref-rule' });
    lineChart(panel(box, t.name, `검증 정확도 · ${t.skill}`), { series, xDomain: [0, 3000], xTicks: [0, 800, 1500, 3000], refs });
  }
  const gain = (v, t) => mean(vals(v, COND.E.f, t)) - mean(vals(v, COND.A.f, t));
  const big = MAIN.map((v) => [v, mean(D.tasks.filter((x) => x.train && x.core).map((x) => gain(v, x.key)))]).sort((a, b) => b[1] - a[1]);
  const nE = new Set(D.runs.filter(COND.E.f).map((r) => r.seed)).size;
  $('#curve-cap').innerHTML = `시험 정확도로 본 800 → 3000걸음 평균 상승(난수 ${nE}개 평균, 학습 과제 6개): `
    + big.map(([v, g]) => `${MODELS[v].name} <b>${g >= 0 ? '+' : '−'}${Math.abs(g * 100).toFixed(1)}%p</b>`).join(' · ')
    + '. 가장 많이 오른 모델이 800걸음 표에서 가장 손해를 본 모델입니다.';
}

// ---------------------------------------------------------------- 5. 관계 사슬 단계 수
function drawHops() {
  const hops = ['chain_hop3', 'chain_hop4', 'chain_hop5'];
  const key = $('#hop-key');
  keyRow(key, MAIN.map((k) => [k, '']));
  refKey(key, [['dash', '끝 개체 찍기 50%'], ['thin', '우연 12.5%']]);
  const box = $('#hop-charts');
  box.innerHTML = '';
  const conds = [
    ['공동 학습 · 800걸음', COND.A.f, '난수 3개 평균'],
    ['관계 사슬만 · 400걸음', (r) => COND.D.f(r) && r.tasks[0] === 'chain', '난수 3개 평균'],
    ['공동 학습 · 3000걸음', COND.E.f, `난수 ${new Set(D.runs.filter(COND.E.f).map((r) => r.seed)).size}개 평균`],
  ];
  for (const [title, f, sub] of conds) {
    const series = MAIN.map((v) => ({
      key: v, dots: true, label: MODELS[v].name,
      points: hops.map((t, i) => [i + 3, mean(vals(v, f, t))]).filter((p) => !Number.isNaN(p[1])),
    }));
    lineChart(panel(box, title, sub), {
      series, xDomain: [2.7, 5.3], xTicks: [3, 4, 5], xFmt: (v) => `${v}단계`, height: 220,
      refs: [{ v: 0.5, cls: 'ref-rule' }, { v: 0.125, cls: 'grid' }],
    });
  }
  const e = (v, t) => mean(vals(v, COND.E.f, t));
  $('#hop-cap').innerHTML = `3000걸음에서 4단계(학습보다 1단계 김) 정확도: `
    + MAIN.map((v) => `${MODELS[v].name} <b>${pct(e(v, 'chain_hop4'))}</b>`).join(' · ')
    + '. 3단계와 4단계가 거의 같으면, 배운 방법이 한 단계 더 긴 사슬에도 그대로 통한다는 뜻입니다.';
}

// ---------------------------------------------------------------- 6. 키-값 지연 사다리
const BUCKETS = ['0-32', '32-64', '64-128', '128-256', '256-512', '512-1024'];
// 256기호까지는 학습과 같은 분포(kvd), 그 뒤는 긴 시험(kvd_long). 128-256 은 두 묶음에 다 있지만 섞지 않는다
const bucketAcc = (delay, i) => (delay[i < 4 ? 'kvd' : 'kvd_long'] || {})[BUCKETS[i]]?.accuracy ?? NaN;
function delayCurve(rs) {
  return BUCKETS.map((_, i) => {
    const per = rs.map((r) => bucketAcc(r.delay, i)).filter((v) => !Number.isNaN(v));
    return [i, per.length ? mean(per) : NaN, per];
  }).filter((p) => !Number.isNaN(p[1]));
}
function drawDelay() {
  const sec = $('#delay-sec');
  const have = ['kvd_all', 'kvd'].filter((t) => D.runs.some((r) => r.delay && r.tasks[0] === t));
  if (!have.length) { sec.hidden = true; return; }
  sec.hidden = false;
  if (!have.includes(delayTrain)) delayTrain = have[0];
  picker($('#delay-picker'), '학습 채점', have.map((t) => [t, t === 'kvd' ? '숨긴 키 질의만 (F)' : '모든 질의 (G)']), delayTrain,
    (k) => { delayTrain = k; drawDelay(); });
  const F = D.runs.filter((r) => r.delay && r.tasks[0] === delayTrain);
  const key = $('#delay-key');
  const combos = [];
  const roster = ['transformer', 'gru', 'folio', 'folio2', 'strata', 'lattice', 'lattice2'];
  for (const v of roster) for (const c of [256, 64]) {
    const rs = F.filter((r) => r.variant === v && r.chunk === c);
    if (rs.length) combos.push({ v, c, rs });
  }
  keyRow(key, roster.filter((v) => combos.some((x) => x.v === v)).map((k) => [k, 'line']));
  refKey(key, [['line', '청크 256 (기울기가 에피소드 전체에 닿음)'], ['thin', '청크 64'], ['dash', '최근 저장값 규칙']]);
  const series = combos.map(({ v, c, rs }) => ({
    key: v, thin: c === 64, points: delayCurve(rs).map(([i, m]) => [i, m]),
    label: c === 256 || v === 'transformer' ? `${MODELS[v].name} ${c}` : null, tipLabel: `${MODELS[v].name} · 청크 ${c} (난수 ${rs.length})`,
  }));
  const base = D.delay_baseline ? BUCKETS.map((_, i) => bucketAcc(D.delay_baseline, i)).filter((v) => !Number.isNaN(v)) : [];
  const box = $('#delay-chart');
  lineChart(box, {
    series, xDomain: [-0.3, BUCKETS.length - 0.7], xTicks: BUCKETS.map((_, i) => i), xFmt: (i) => BUCKETS[i], height: 280,
    refs: [{ v: mean(base), cls: 'ref-rule' }, { v: 0.125, cls: 'grid' }],
    vlines: [{ x: 1.5, label: '64기호: 창 · 청크 64 경계' }, { x: 3.5, label: '학습 최대 지연' }],
  });
  const at = (v, c, b) => { const cc = combos.find((x) => x.v === v && x.c === c); if (!cc) return '–'; const p = delayCurve(cc.rs).find((q) => BUCKETS[q[0]] === b); return p ? pct(p[1]) : '–'; };
  $('#delay-cap').innerHTML = `지연 128–256기호(학습 범위 안, 청크 64 밖): `
    + combos.map(({ v, c }) => `${MODELS[v].name} ${c} <b>${at(v, c, '128-256')}</b>`).join(' · ')
    + `. 지연 256–512기호(학습보다 김): ` + combos.map(({ v, c }) => `${MODELS[v].name} ${c} <b>${at(v, c, '256-512')}</b>`).join(' · ')
    + '. 우연 12.5%. 시험은 두 학습 방식 모두 숨긴 키 질의만 채점합니다. 256기호까지는 학습과 같은 분포의 시험(길이 256), 그 뒤는 길이 1024 시험입니다. 2000걸음 · 난수 3개 평균.';
}

// ---------------------------------------------------------------- 7. 새 과제 묶음 (계획 H): 난이도 손잡이별 곡선
// 손잡이 값 하나 = 점 하나. 학습 범위는 학습 과제의 시험에서, 그 너머는 '더 어렵게' 시험에서 가져온다.
const FAMILIES = {
  cyc: { title: '순환 사슬', sub: '걸음 수 k', xs: [1, 2, 3, 4, 5, 6, 7], src: (k) => [k <= 3 ? 'cyc' : 'cyc_far', String(k)], fmt: (k) => `${k}`, split: 3.5 },
  count: { title: '셈 나머지', sub: '나머지 m · 왼쪽 256기호, 오른쪽 1024기호', xs: [0, 1, 2, 3, 4, 5], src: (i) => [i < 3 ? 'count' : 'count_long', String([2, 3, 5][i % 3])], fmt: (i) => `mod ${[2, 3, 5][i % 3]}`, split: 2.5, splitLabel: '1024기호 →' },
  copy: { title: '베끼기 · 뒤집기', sub: '원문 길이 n', xs: [...Array(21).keys()].map((i) => i + 4).concat([...Array(21).keys()].map((i) => i + 28)), src: (n) => [n <= 24 ? 'copy' : 'copy_long', String(n)], fmt: (n) => `${n}`, ticks: [4, 12, 24, 32, 40, 48], split: 26 },
  paren: { title: '괄호 깊이', sub: '연달아 닫기 전 깊이', xs: [1, 2, 3, 4, 5, 6, 7], src: (d) => [d <= 3 ? 'paren' : 'paren_deep', String(d)], fmt: (d) => `${d}`, split: 3.5 },
  slowp: { title: '느린 주기', sub: '주기 P · 오프셋이 바뀌는 자리만', xs: [0, 1, 2, 3], src: (i) => [i < 3 ? 'slowp' : 'slowp_long', String([32, 64, 128, 256][i])], fmt: (i) => `P ${[32, 64, 128, 256][i]}`, split: 2.5 },
};
const isSuite = (r) => !!r.knobs && r.tasks[0] === 'cyc';
function knobCurve(rs, fam) {
  return fam.xs.map((x) => {
    const [t, k] = fam.src(x);
    return [x, mean(rs.map((r) => (r.knobs[t] && r.knobs[t][k] ? r.knobs[t][k].accuracy : NaN)).filter((v) => !Number.isNaN(v)))];
  }).filter((p) => !Number.isNaN(p[1]));
}
function drawSuite() {
  const sec = $('#suite-sec');
  const H = D.runs.filter(isSuite);
  if (!H.length) { sec.hidden = true; return; }
  sec.hidden = false;
  const roster = SUITE_ROSTER.filter((v) => H.some((r) => r.variant === v));
  keyRow($('#suite-key'), roster.map((k) => [k, 'line']));
  refKey($('#suite-key'), [['dash', '단순 규칙'], ['thin', 'Strata 모든 층 매번 갱신 (느린 주기만)']]);
  const box = $('#suite-charts');
  box.innerHTML = '';
  for (const [fk, fam] of Object.entries(FAMILIES)) {
    const series = roster.map((v) => {
      const rs = H.filter((r) => r.variant === v);
      return { key: v, points: knobCurve(rs, fam), label: MODELS[v].name, tipLabel: `${MODELS[v].name} (난수 ${rs.length})` };
    });
    if (fk === 'slowp') {
      const rs = H.filter((r) => r.variant === 'strata_sameperiod');
      if (rs.length) series.push({ key: 'strata_sameperiod', thin: true, points: knobCurve(rs, fam), tipLabel: 'Strata 모든 층 매번 갱신' });
    }
    if (D.knob_baseline) {
      const pts = fam.xs.map((x) => { const [t, k] = fam.src(x); const b = D.knob_baseline[t]; return [x, b && b[k] ? b[k].accuracy : NaN]; })
        .filter((q) => !Number.isNaN(q[1]));
      series.push({ key: 'rule', dash: true, points: pts, tipLabel: `규칙: ${D.knob_baseline[fk].name}` });
    }
    const lo = fam.xs[0], hi = fam.xs[fam.xs.length - 1], pad = (hi - lo) * 0.04 || 0.3;
    lineChart(panel(box, fam.title, `${fam.sub} · ${TASK[fk].skill}`), {
      series, xDomain: [lo - pad, hi + pad], xTicks: fam.ticks || fam.xs, xFmt: fam.fmt, height: 250,
      refs: [{ v: TASK[fk].chance, cls: 'grid' }], vlines: [{ x: fam.split, label: fam.splitLabel || '학습보다 어렵게 →' }],
    });
  }
  // 과제별 전체 정확도 (시험 10개)
  const box2 = $('#suite-acc');
  box2.innerHTML = '';
  for (const t of D.tasks.filter((x) => x.suite)) {
    const rows = roster.map((v) => ({ key: v, label: nameOf(v), values: vals(v, isSuite, t.key) }));
    dotRows(panel(box2, t.name, `${t.skill}${t.train ? '' : ' · 시험만'}`), { rows, domain: [0, 1], refs: refsFor(t.key) });
  }
  const hard = { cyc: 7, copy: 48, paren: 7 };
  const at = (v, fk) => { const p = knobCurve(H.filter((r) => r.variant === v), FAMILIES[fk]).find((q) => q[0] === hard[fk]); return p ? pct(p[1]) : '–'; };
  $('#suite-cap').innerHTML = `가장 어려운 칸. 7걸음: ` + roster.map((v) => `${MODELS[v].name} <b>${at(v, 'cyc')}</b>`).join(' · ')
    + `. 원문 48기호: ` + roster.map((v) => `${MODELS[v].name} <b>${at(v, 'copy')}</b>`).join(' · ')
    + `. 깊이 7: ` + roster.map((v) => `${MODELS[v].name} <b>${at(v, 'paren')}</b>`).join(' · ')
    + `. 공동 학습 · 3000걸음 · 난수 ${new Set(H.map((r) => r.seed)).size}개 평균. 세로선 왼쪽이 학습 범위, 오른쪽이 학습보다 어려운 시험입니다. 가는 가로선은 우연 수준입니다.`;
}

// ---------------------------------------------------------------- 8. 개선판: 원래 → 개선
function dumbbells(container, { rows, color }) {
  const rowH = 22, top = 22, bottom = 24, left = 150, right = 16;
  const height = top + rows.length * rowH + bottom;
  const { svg, width } = makeSvg(container, height);
  const x = (v) => left + v * (width - left - right);
  for (const v of [0, 0.25, 0.5, 0.75, 1]) {
    el('line', { x1: x(v), x2: x(v), y1: top - 4, y2: height - bottom, class: 'grid' }, svg);
    text(svg, x(v), height - bottom + 14, `${v * 100}`, { 'text-anchor': 'middle', class: 't-num' });
  }
  rows.forEach((r, i) => {
    const cy = top + i * rowH + rowH / 2;
    text(svg, left - 10, cy + 4, r.label, { 'text-anchor': 'end', class: 't-ink' });
    if (Number.isNaN(r.a) || Number.isNaN(r.b)) { text(svg, left + 4, cy + 4, '실행 없음', { class: 't-ink2' }); return; }
    el('line', { x1: x(r.a), x2: x(r.b), y1: cy, y2: cy, style: `stroke:${color};stroke-width:2;stroke-opacity:0.5` }, svg);
    el('circle', { cx: x(r.a), cy, r: 4.5, style: `fill:var(--page);stroke:${color};stroke-width:1.5` }, svg);
    el('circle', { cx: x(r.b), cy, r: 5, style: `fill:${color};stroke:var(--page);stroke-width:1.5` }, svg);
    const hit = el('rect', { x: 0, y: cy - rowH / 2, width, height: rowH, class: 'hit' }, svg);
    hit.addEventListener('mousemove', (ev) => showTip(ev, r.label, [['원래', pct(r.a)], ['개선판', pct(r.b), color],
      ['차이', `${r.b >= r.a ? '+' : '−'}${Math.abs((r.b - r.a) * 100).toFixed(1)}%p`]]));
    hit.addEventListener('mouseleave', hideTip);
  });
}
function improvedRows(v1, v2) {
  const E = COND.E.f;
  const F = (r) => r.delay && r.tasks[0] === 'kvd' && r.chunk === 256;
  const rows = [];
  const core = D.tasks.filter((t) => t.core && t.key !== 'chain_long').map((t) => t.key).concat(['chain_hop4']);
  for (const t of core) rows.push({ label: TASK[t].name, a: mean(vals(v1, E, t)), b: mean(vals(v2, E, t)) });
  for (const [i, b] of [[3, '128–256'], [4, '256–512']]) {
    const g = (v) => mean(runsOf(v, F).map((r) => bucketAcc(r.delay, i)).filter((q) => !Number.isNaN(q)));
    rows.push({ label: `지연 사다리 ${b}`, a: g(v1), b: g(v2) });
  }
  for (const t of D.tasks.filter((x) => x.suite).map((x) => x.key)) rows.push({ label: TASK[t].name, a: mean(vals(v1, isSuite, t)), b: mean(vals(v2, isSuite, t)) });
  return rows;
}
function drawImproved() {
  const sec = $('#v2-sec');
  const have = IMPROVED.filter((v) => D.runs.some((r) => r.variant === v));
  if (!have.length) { sec.hidden = true; return; }
  sec.hidden = false;
  const box = $('#v2-charts');
  box.innerHTML = '';
  const caps = [];
  for (const v2 of have) {
    const v1 = MODELS[v2].v1;
    const rows = improvedRows(v1, v2);
    dumbbells(panel(box, `${MODELS[v1].name} → ${MODELS[v2].name}`, '빈 원 = 원래 · 찬 원 = 개선판 · 정확도 %'), { rows, color: colorOf(v2) });
    const ok = rows.filter((r) => !Number.isNaN(r.a) && !Number.isNaN(r.b));
    const up = ok.filter((r) => r.b - r.a > 0.02).length, down = ok.filter((r) => r.a - r.b > 0.02).length;
    caps.push(`${MODELS[v2].name}: 비교 ${ok.length}개 중 <b>${up}개 오름</b> · ${down}개 내림(±2%p 넘는 것) · 평균 ${(mean(ok.map((r) => r.b - r.a)) * 100).toFixed(1)}%p`);
  }
  $('#v2-cap').innerHTML = caps.join('. ') + '. 위 6줄 + 관계 사슬 4단계는 공동 학습 3000걸음, 지연 사다리는 청크 256 · 2000걸음, 나머지는 새 과제 묶음. 모두 난수 3개 평균.';
}

// ---------------------------------------------------------------- 9. 대조군
function drawAblations() {
  const opts = ['chain', 'chain_hop3', 'mixed', 'kv', 'switch_fast', 'cycle'].map((t) => [t, TASK[t].name]);
  picker($('#abl-picker'), '문제', opts, ablTask, (k) => { ablTask = k; drawAblations(); });
  const box = $('#abl-charts');
  box.innerHTML = '';
  const caps = [];
  for (const base of CANDS) {
    const vs = [base, ...Object.keys(MODELS).filter((k) => MODELS[k].base === base)];
    const rows = vs.map((v) => ({ key: v, label: v === base ? '원래' : MODELS[v].name, values: vals(v, COND.A.f, ablTask) }));
    const bm = mean(rows[0].values);
    const refs = [{ v: bm, label: '원래', cls: 'ref' }, ...refsFor(ablTask).filter((r) => r.cls !== 'ref')];
    dotRows(panel(box, MODELS[base].name, '공동 학습 · 지속 기억 · 800걸음'), { rows, domain: [0, 1], refs });
    for (const r of rows.slice(1)) caps.push(`${nameOf(r.key)} ${p1(bm)} → <b>${p1(mean(r.values))}</b>`);
  }
  $('#abl-cap').innerHTML = `<b>${TASK[ablTask].name}</b>: ` + caps.join(' · ') + ' (%, 난수 3개 평균).';
}

// ---------------------------------------------------------------- 7. 비용
function drawCosts() {
  const M = D.measure;
  if (!M) { $('#cost-cap').textContent = '자원 측정 결과가 없습니다 (py main.py 가 측정합니다).'; return; }
  const rows = M.rows.slice().sort((a, b) => Object.keys(MODELS).indexOf(a.variant) - Object.keys(MODELS).indexOf(b.variant));
  const box = $('#cost-chart');
  box.innerHTML = '<div class="multiples" id="cost-multi"></div>';
  const multi = $('#cost-multi');
  const logRows = (container, getter, fmt, ticks, max) => {
    const rowH = 22, top = 8, bottom = 24, left = 150, right = 70;
    const height = top + rows.length * rowH + bottom;
    const { svg, width } = makeSvg(container, height);
    const lo = Math.log10(ticks[0]), hi = Math.log10(max);
    const x = (v) => left + ((Math.log10(v) - lo) / (hi - lo)) * (width - left - right);
    for (const t of ticks) {
      el('line', { x1: x(t), x2: x(t), y1: top, y2: height - bottom, class: 'grid' }, svg);
      text(svg, x(t), height - bottom + 14, fmt(t), { 'text-anchor': 'middle', class: 't-num' });
    }
    rows.forEach((r, i) => {
      const cy = top + i * rowH + rowH / 2, v = getter(r), c = colorOf(r.variant), ring = !!MODELS[r.variant].base;
      text(svg, left - 10, cy + 4, ring ? `  ${MODELS[r.variant].name}` : nameOf(r.variant), { 'text-anchor': 'end', class: ring ? 't-ink2' : 't-ink' });
      el('line', { x1: x(ticks[0]), x2: x(v), y1: cy, y2: cy, style: `stroke:${c};stroke-width:2;stroke-opacity:0.35` }, svg);
      el('circle', { cx: x(v), cy, r: 5, style: ring ? `fill:var(--page);stroke:${c};stroke-width:2` : `fill:${c};stroke:var(--page);stroke-width:2` }, svg);
      text(svg, x(v) + 9, cy + 4, fmt(v), { class: 't-num t-ink2' });
    });
  };
  logRows(panel(multi, '학습 한 걸음', '밀리초 · 로그 눈금 · 짧을수록 좋음'), (r) => r.train_sec_per_step * 1000,
    (v) => (v >= 1000 ? `${+(v / 1000).toFixed(1)}초` : `${Math.round(v)}ms`), [1, 10, 100, 1000], 3000);
  logRows(panel(multi, '학습 피크 메모리', 'GPU · 로그 눈금'), (r) => r.train_peak_bytes / 2 ** 20,
    (v) => (v >= 1000 ? `${+(v / 1024).toFixed(1)}GiB` : `${Math.round(v)}MiB`), [10, 100, 1000], 3000);
  const tf = rows.find((r) => r.variant === 'transformer');
  const ratios = CANDS.map((v) => rows.find((r) => r.variant === v)).filter(Boolean).map((r) => r.train_sec_per_step / tf.train_sec_per_step);
  $('#cost-cap').innerHTML = `측정 ${M.measured_at}. 후보 세 개의 학습 한 걸음은 Transformer 의 <b>${Math.min(...ratios).toFixed(0)}–${Math.max(...ratios).toFixed(0)}배</b>입니다. 빈 원은 대조군입니다.`;
  const head = '<thead><tr><th class="l">모델</th><th>파라미터</th><th>학습 ms/걸음</th><th>학습 기호/초</th><th>학습 피크 메모리</th><th>지연 GPU<span class="sub">평균 / 90%</span></th><th>지연 CPU<span class="sub">평균 / 90%</span></th><th>들고 다니는 상태<span class="sub">기록 1개</span></th></tr></thead>';
  $('#cost-table').innerHTML = head + '<tbody>' + rows.map((r) => `<tr>
    <td class="l">${MODELS[r.variant].base ? `<span class="dim">${MODELS[MODELS[r.variant].base].name}</span> ${MODELS[r.variant].name}` : `<b>${nameOf(r.variant)}</b>`}</td>
    <td class="n">${num(r.params)}</td><td class="n">${num(Math.round(r.train_sec_per_step * 1000))}</td>
    <td class="n">${num(Math.round(r.train_tokens_per_sec))}</td><td class="n">${bytes(r.train_peak_bytes)}</td>
    <td class="n">${r.latency.cuda.mean_ms.toFixed(2)} / ${r.latency.cuda.p90_ms.toFixed(2)} ms</td>
    <td class="n">${r.latency.cpu.mean_ms.toFixed(2)} / ${r.latency.cpu.p90_ms.toFixed(2)} ms</td>
    <td class="n">${bytes(r.state_bytes_batch1)}${r.variant.startsWith('transformer') ? `<span class="sub">최근 기호 ${r.variant === 'transformer' ? 64 : 256}개</span>` : ''}</td></tr>`).join('') + '</tbody>';
}

// ---------------------------------------------------------------- 8. 검증
function drawChecks() {
  const C = D.checks;
  if (!C) { $('#check-cap').textContent = '검증 결과가 없습니다 (py main.py 가 검증합니다).'; return; }
  const checks = [...new Set(C.rows.filter((r) => r.variant).map((r) => r.check))];
  const variants = [...new Set(C.rows.filter((r) => r.variant).map((r) => r.variant))];
  const NAMES = { 'forward = step 누적': '한 번에 = 한 개씩', '인과성': '인과성', '상태 누출 없음': '누출 없음', '청크 분할 일치': '청크 분할', '유한 손실·gradient': '유한 손실·기울기', '작은 결정적 문제 학습': '작은 문제 학습' };
  const cell = (r) => (r ? `<td title="${r.msg.replace(/"/g, '&quot;')}">${r.ok ? '<span class="ok">✓ 통과</span>' : '<span class="fail">✗ 실패</span>'}</td>` : '<td class="dim">–</td>');
  $('#check-table').innerHTML = '<thead><tr><th class="l">모델</th>' + checks.map((c) => `<th>${NAMES[c] || c}</th>`).join('') + '</tr></thead><tbody>'
    + variants.map((v) => `<tr><td class="l">${MODELS[v] ? nameOf(v) : v}</td>` + checks.map((c) => cell(C.rows.find((r) => r.variant === v && r.check === c))).join('') + '</tr>').join('')
    + '</tbody>';
  const gen = C.rows.find((r) => !r.variant);
  const failed = C.rows.filter((r) => !r.ok).length;
  $('#check-cap').innerHTML = `검사 ${C.checked_at} · ${C.device} · ${C.dtype}. 전체 ${C.rows.length}개 중 실패 <b>${failed}</b>개. `
    + `과제 생성기: ${gen ? (gen.ok ? '모든 질의의 답을 따로 다시 계산해 일치 확인' : '실패 — ' + gen.msg) : '–'}. 칸에 마우스를 올리면 잰 값이 보입니다.`;
}

// ---------------------------------------------------------------- 요약 (숫자는 자료에서 계산)
function drawFindings() {
  const A = COND.A.f, E = COND.E.f;
  const m = (v, f, t) => mean(vals(v, f, t));
  const trainTasks = D.tasks.filter((t) => t.train && t.core).map((t) => t.key);
  const best = (f) => trainTasks.map((t) => [t, MAIN.slice().sort((a, b) => m(b, f, t) - m(a, f, t))[0]]);
  const nE = new Set(D.runs.filter(E).map((r) => r.seed)).size;
  const items = [];
  items.push(`<b>800걸음 비교는 모두 덜 배운 상태였습니다.</b> 같은 조건으로 3000걸음까지 늘리자 Transformer 의 키-값 정확도가 ${pct(m('transformer', A, 'kv'))} → <b>${pct(m('transformer', E, 'kv'))}</b>로 뛰었습니다(난수 ${nE}개 평균). `
    + (new Set(best(E).map(([, v]) => v)).size === 1 ? `3000걸음에서는 학습 과제 ${trainTasks.length}개 모두 ${MODELS[best(E)[0][1]].name} 가 1위입니다.`
      : '3000걸음에서 문제별 1위: ' + best(E).map(([t, v]) => `${TASK[t].name} ${MODELS[v].name}`).join(' · ') + '.'));
  items.push(`<b>순환형이 앞선 것은 학습보다 긴 관계 사슬입니다.</b> 3000걸음에서 한 단계 긴 4단계 사슬: GRU ${pct(m('gru', E, 'chain_hop4'))}, <span class="m-folio">Folio</span> ${pct(m('folio', E, 'chain_hop4'))}, <span class="m-strata">Strata</span> ${pct(m('strata', E, 'chain_hop4'))}, Transformer ${pct(m('transformer', E, 'chain_hop4'))} (끝 개체 찍기 50%).`);
  const avg = (v) => mean(trainTasks.map((t) => m(v, E, t)));
  items.push(`<b>세 후보 중 GRU 를 넘은 것은 없습니다.</b> 3000걸음 공동 학습, 학습 과제 6개 평균: Transformer ${pct(avg('transformer'))} · GRU ${pct(avg('gru'))} · <span class="m-folio">Folio</span> ${pct(avg('folio'))} · <span class="m-strata">Strata</span> ${pct(avg('strata'))} · <span class="m-lattice">Lattice</span> ${pct(avg('lattice'))}.`);
  const have = IMPROVED.filter((v) => D.runs.some((r) => r.variant === v));
  if (have.length) {
    const gain = (v2) => { const ok = improvedRows(MODELS[v2].v1, v2).filter((r) => !Number.isNaN(r.a) && !Number.isNaN(r.b)); return mean(ok.map((r) => r.b - r.a)); };
    items.push(`<b>고친 것 중 효과가 분명한 것은 Lattice2 하나입니다.</b> 입력을 모든 칸에 넣고 메시지를 정규화하자 난수에 따라 무너지던 학습이 안정됐습니다. 관계 사슬 ${pct(m('lattice', E, 'chain'))} ± ${pct(sd(vals('lattice', E, 'chain')))} → <b>${pct(m('lattice2', E, 'chain'))}</b> ± ${pct(sd(vals('lattice2', E, 'chain')))}, 비교 전체 평균 ${(gain('lattice2') * 100).toFixed(1)}%p. Folio2 의 쓰기 세기 · 내용 주소는 평균 ${(gain('folio2') * 100).toFixed(1)}%p 로 차이가 없었습니다. 그래도 Lattice2 는 GRU 아래입니다.`);
  }
  const H = D.runs.filter(isSuite);
  if (H.length) {
    const kn = (v, t, k) => mean(H.filter((r) => r.variant === v).map((r) => (r.knobs[t] && r.knobs[t][k] ? r.knobs[t][k].accuracy : NaN)).filter((q) => !Number.isNaN(q)));
    const roster = SUITE_ROSTER.filter((v) => H.some((r) => r.variant === v));
    const cycBest = Math.max(...roster.map((v) => m(v, isSuite, 'cyc')));
    items.push(`<b>새 과제에서는 각자 잘하는 것이 갈렸습니다.</b> 창 안의 느린 주기(P 64, 오프셋이 바뀌는 자리)는 Transformer 가 ${pct(kn('transformer', 'slowp', '64'))}, GRU · Folio · Lattice 는 ${pct(Math.max(...['gru', 'folio', 'lattice'].map((v) => kn(v, 'slowp', '64'))))} 이하입니다. 베끼기(원문 4기호)는 GRU ${pct(kn('gru', 'copy', '4'))} · <span class="m-folio">Folio</span> ${pct(kn('folio', 'copy', '4'))}, Transformer ${pct(kn('transformer', 'copy', '4'))}. 끝 개체가 없는 순환 사슬은 모든 모델이 우연(12.5%) 수준입니다(최고 ${pct(cycBest)}).`);
    items.push(`<b>Strata 의 '층마다 다른 속도'는 약한 신호만 보였습니다.</b> 느린 주기 P 64 의 바뀌는 자리에서 <span class="m-strata">Strata</span> ${pct(kn('strata', 'slowp', '64'))}, 모든 층을 매번 갱신한 대조군 ${pct(kn('strata_sameperiod', 'slowp', '64'))}. 다른 순환형보다 높지만 풀었다고 할 수준은 아닙니다. 대신 베끼기에서는 대조군보다 낮습니다(원문 4기호 ${pct(kn('strata', 'copy', '4'))} 대 ${pct(kn('strata_sameperiod', 'copy', '4'))}).`);
  }
  const F = D.runs.filter((r) => r.delay && r.tasks[0] === 'kvd');
  const fd = (v, c, i) => mean(F.filter((r) => r.variant === v && r.chunk === c).map((r) => bucketAcc(r.delay, i)));
  if (F.length) {
    items.push(`<b>64기호 밖 기억을 막은 것은 구조가 아니라 학습 방식이었습니다.</b> 지연 사다리에서 기울기를 에피소드 전체에 흘리자(청크 256), 학습보다 긴 지연 256–512기호에서 GRU ${pct(fd('gru', 64, 4))} → <b>${pct(fd('gru', 256, 4))}</b>, <span class="m-folio">Folio</span> ${pct(fd('folio', 64, 4))} → <b>${pct(fd('folio', 256, 4))}</b>. <span class="m-strata">Strata</span> ${pct(fd('strata', 256, 4))} · Folio2 ${pct(fd('folio2', 256, 4))} 도 GRU 와 비슷해, 주소 기억 구조의 이점은 여기서도 보이지 않았습니다.`);
  }
  const G = D.runs.filter((r) => r.delay && r.tasks[0] === 'kvd_all');
  const gd = (v, c, i) => mean(G.filter((r) => r.variant === v && r.chunk === c).map((r) => bucketAcc(r.delay, i)));
  if (G.length && F.length) {
    items.push(`<b>모델마다 맞는 학습 신호가 다릅니다.</b> 방해 구간 질의까지 채점하자 Transformer(창 64)는 창 안 32–64기호에서 ${pct(fd('transformer', 64, 1))} → <b>${pct(gd('transformer', 64, 1))}</b>로 올랐습니다. 같은 채점에서 GRU 는 ${pct(fd('gru', 256, 1))} → ${pct(gd('gru', 256, 1))}, <span class="m-folio">Folio</span> 는 ${pct(fd('folio', 256, 1))} → ${pct(gd('folio', 256, 1))}로 떨어졌습니다(청크 256). 최근 키만 좇게 된 것입니다.`);
  }
  if (D.measure) {
    const r = (v) => D.measure.rows.find((x) => x.variant === v);
    const ratios = CANDS.map((v) => r(v).train_sec_per_step / r('transformer').train_sec_per_step);
    items.push(`<b>비용:</b> 한 모델씩 보통 방식으로 재면 후보의 학습 한 걸음은 Transformer 의 ${Math.min(...ratios).toFixed(0)}–${Math.max(...ratios).toFixed(0)}배 걸립니다. 난수 3개를 묶어 한 번에 계산하고 CUDA graph 로 재생하는 학습기로 실험 전체를 다시 돌릴 수 있게 됐지만, 구조 자체의 비용은 그대로입니다.`);
  }
  $('#findings').innerHTML = items.map((s) => `<li>${s}</li>`).join('');
}

// ---------------------------------------------------------------- 전체
function drawAll() {
  drawSchematics();
  drawFindings();
  drawTasks();
  drawAccuracy();
  drawCurves();
  drawHops();
  drawDelay();
  drawSuite();
  drawImproved();
  drawAblations();
  drawCosts();
  drawChecks();
}

function setupTheme() {
  const btn = $('#theme'), order = ['system', 'light', 'dark'], names = { system: '시스템', light: '밝게', dark: '어둡게' };
  let cur = 'system';
  try { cur = localStorage.getItem('theme') || 'system'; } catch (e) { /* 저장소를 못 쓰면 시스템 설정 */ }
  const apply = () => {
    if (cur === 'system') document.documentElement.removeAttribute('data-theme');
    else document.documentElement.setAttribute('data-theme', cur);
    btn.textContent = `테마: ${names[cur]}`;
  };
  btn.addEventListener('click', () => {
    cur = order[(order.indexOf(cur) + 1) % order.length];
    try { localStorage.setItem('theme', cur); } catch (e) { /* 무시 */ }
    apply();
  });
  apply();
}

async function main() {
  setupTheme();
  const st = $('#status');
  try {
    // 내보낸 정적 페이지(export_site.py)는 RESULTS_URL 을 정해 두고, 로컬 서버에서는 API 를 읽는다
    const res = await fetch(window.RESULTS_URL || '/api/results');
    D = await res.json();
    if (D.error) throw new Error(D.error);
  } catch (err) {
    st.textContent = `결과를 불러오지 못했습니다: ${err.message}.` + (window.RESULTS_URL ? '' : ' py main.py 로 서버를 띄웠는지 확인하세요.');
    st.className = 'error';
    return;
  }
  for (const t of D.tasks) TASK[t.key] = t;
  for (const v of MAIN.concat(D.ablations, IMPROVED)) {
    const p = $(`#p-${v}`);
    if (p) p.textContent = num(D.models[v].params);
    const mr = D.measure && D.measure.rows.find((r) => r.variant === v);
    const s = $(`#s-${v}`);
    if (s && mr) s.textContent = bytes(mr.state_bytes_batch1);
  }
  const nTrain = D.tasks.filter((t) => t.train && (t.core || t.suite)).length;
  $('#by-tasks').textContent = `학습 ${nTrain} · 시험만 ${D.tasks.filter((t) => !t.train).length}`;
  const planned = Object.values(D.plan).reduce((s, g) => s + g.total, 0), done = Object.values(D.plan).reduce((s, g) => s + g.done, 0);
  $('#by-runs').textContent = `${done}/${planned}개`;
  st.remove();
  $('#foot').innerHTML = `자료 생성 ${D.generated_at} · 실행 ${D.runs.length}개 · 계획 ` + Object.entries(D.plan).map(([g, p]) => `${g} ${p.done}/${p.total}`).join(' · ')
    + (D.measure ? ` · 자원 측정 ${D.measure.measured_at}` : '') + ' · 모든 수치는 results/candidates 의 실행 결과에서 화면이 직접 계산합니다.';
  drawAll();
  let w = innerWidth;
  addEventListener('resize', () => { if (Math.abs(innerWidth - w) > 30) { w = innerWidth; drawAll(); } });
}

main();
