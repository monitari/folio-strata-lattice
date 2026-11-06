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
  A: { label: '800걸음', f: (r) => r.mode === 'joint' && r.memory === 'persistent' && r.steps === 800 },
  B: { label: '800걸음 · 64기호마다 기억 지움', f: (r) => r.mode === 'joint' && r.memory === 'window' && r.steps === 800 },
  D: { label: '과제 하나씩 · 400걸음', f: (r) => r.mode === 'separate' && r.memory === 'persistent' && r.steps === 400 },
  E: { label: '3000걸음', f: (r) => r.mode === 'joint' && r.memory === 'persistent' && r.steps === 3000 },
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
let cond = 'E', metric = 'accuracy', ablTask = 'chain', delayTrain = 'kvd';

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
    const mk = MODELS[row.key].base || row.key;
    const rowG = el('g', { 'data-m': mk }, svg);
    text(rowG, left - 10, cy + 4, row.label, { 'text-anchor': 'end', class: 't-ink' });
    const c = colorOf(row.key), ring = !!MODELS[row.key].base;
    if (!row.values.length) {
      text(rowG, left + 4, cy + 4, '실행 없음', { class: 't-ink2' });
      return;
    }
    const m = mean(row.values), s = sd(row.values);
    for (const v of row.values) {
      el('circle', { cx: x(v), cy, r: 3, style: `fill:${c};fill-opacity:0.35` }, rowG);
    }
    if (row.values.length > 1) {
      el('line', { x1: x(Math.max(lo, m - s)), x2: x(Math.min(hi, m + s)), y1: cy, y2: cy, style: `stroke:${c};stroke-width:2` }, rowG);
    }
    el('circle', { cx: x(m), cy, r: 5.5, style: ring ? `fill:var(--page);stroke:${c};stroke-width:2` : `fill:${c};stroke:var(--page);stroke-width:2` }, rowG);
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
    const mk = MODELS[s.key].base || s.key;
    el('path', { d, fill: 'none', 'data-m': mk, style: `stroke:${c};stroke-width:${s.thin ? 1.25 : 2};stroke-opacity:${s.thin ? 0.55 : 1};stroke-linejoin:round${s.dash ? ';stroke-dasharray:4 3' : ''}` }, svg);
    if (s.dots) for (const [a, b] of s.points) el('circle', { cx: x(a), cy: y(b), r: 3.5, 'data-m': mk, style: `fill:${c};stroke:var(--page);stroke-width:1.5` }, svg);
    if (s.label) { const [a, b] = s.points[s.points.length - 1]; ends.push({ y: y(b), x: x(a), label: s.label, mk }); }
  }
  // 끝 이름표: 겹치지 않게 아래로 밀고, 넘치면 위로 당긴다
  ends.sort((a, b) => a.y - b.y);
  for (let i = 1; i < ends.length; i++) ends[i].y = Math.max(ends[i].y, ends[i - 1].y + 12);
  const maxY = height - bottom;
  for (let i = ends.length - 1; i >= 0; i--) {
    if (ends[i].y > maxY) ends[i].y = maxY;
    if (i < ends.length - 1 && ends[i].y > ends[i + 1].y - 12) ends[i].y = ends[i + 1].y - 12;
  }
  for (const e of ends) text(svg, e.x + 6, e.y + 4, e.label, { class: 't-ink2', 'data-m': e.mk });
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

// ---------------------------------------------------------------- 모델 강조 (고정 막대의 버튼 → 모든 그림)
const FOCUS_ROSTER = ['transformer', 'transformer256', 'gru', 'folio', 'folio2', 'strata', 'lattice', 'lattice2'];
const focus = new Set();
function drawChips() {
  const box = $('#chips');
  box.innerHTML = '<span class="lab">강조</span>';
  for (const v of FOCUS_ROSTER) {
    const b = document.createElement('button');
    b.type = 'button';
    b.setAttribute('aria-pressed', String(focus.has(v)));
    b.append(swatch(v), MODELS[v].name);
    b.addEventListener('click', () => { if (focus.has(v)) focus.delete(v); else focus.add(v); applyFocus(); drawChips(); });
    box.appendChild(b);
  }
}
function applyFocus() {
  let st = $('#focus-style');
  if (!st) { st = document.createElement('style'); st.id = 'focus-style'; document.head.appendChild(st); }
  st.textContent = focus.size ? `[data-m]{opacity:.13}${[...focus].map((v) => `[data-m="${v}"]{opacity:1}`).join('')}` : '';
}

// ---------------------------------------------------------------- 결과: 큰 숫자
function drawTiles() {
  const E = COND.E.f, A = COND.A.f;
  const m = (v, f, t) => mean(vals(v, f, t));
  const core = D.tasks.filter((t) => t.train && t.core).map((t) => t.key);
  const avg = (v) => mean(core.map((t) => m(v, E, t)));
  const cands = CANDS.concat(IMPROVED).sort((a, b) => avg(b) - avg(a));
  const F = D.runs.filter((r) => r.delay && r.tasks[0] === 'kvd');
  const fd = (v, c, i) => mean(F.filter((r) => r.variant === v && r.chunk === c).map((r) => bucketAcc(r.delay, i)));
  const r = (v) => D.measure.rows.find((x) => x.variant === v);
  const ratios = CANDS.map((v) => r(v).train_sec_per_step / r('transformer').train_sec_per_step);
  const tiles = [
    ['#learn', pct(m('transformer', E, 'kv')), 'Transformer, 키-값', `800걸음 때는 ${pct(m('transformer', A, 'kv'))}였다`],
    ['#delay-sec', pct(fd('gru', 256, 4)), 'GRU, 256–512기호 전 기억', `학습 신호를 64기호에서 끊으면 ${pct(fd('gru', 64, 4))}`],
    ['#board', `${p1(avg('gru'))} : ${p1(avg(cands[0]))}`, `GRU 대 최고 후보 ${MODELS[cands[0]].name}`, `학습 과제 ${core.length}개 평균 정확도(%)`],
    ['#cost', `${Math.min(...ratios).toFixed(0)}–${Math.max(...ratios).toFixed(0)}배`, '후보 학습 한 걸음이 더 걸린다', 'Transformer와 비교'],
  ];
  $('#tiles').innerHTML = tiles.map(([h, big, what, sub]) => `<a class="tile" href="${h}"><span class="big">${big}</span><span class="what">${what}</span><span class="sub">${sub}</span></a>`).join('');
}

// ---------------------------------------------------------------- 결과: 점수판 (열 하나 = 시험 하나)
function boardColumns() {
  const E = COND.E.f;
  const fromTask = (t, f, label) => ({
    label: label || TASK[t].name, values: (v) => vals(v, f, t), oracle: oracleOf(t), chance: TASK[t].chance,
    rule: TASK[t].rule ? TASK[t].rule.accuracy : null, ruleName: TASK[t].rule ? TASK[t].rule.name : null,
  });
  const fromBucket = (i, label) => ({
    label,
    values: (v) => D.runs.filter((r) => r.delay && r.tasks[0] === 'kvd' && r.variant === v && r.chunk === (v === 'transformer' ? 64 : 256))
      .map((r) => bucketAcc(r.delay, i)).filter((q) => !Number.isNaN(q)),
    oracle: 1, chance: 0.125, rule: D.delay_baseline ? bucketAcc(D.delay_baseline, i) : null, ruleName: '가장 최근 저장값',
  });
  return [
    { title: '함께 학습 · 3000걸음', href: '#learn', cols: [fromTask('switch_fast', E, '요리 교대'), fromTask('phrase', E, '코스'), fromTask('cycle', E, '24접시'), fromTask('kv', E, '키-값'), fromTask('mixed', E, '혼합'), fromTask('chain', E, '관계 사슬')], avg: true },
    { title: '긴 사슬', href: '#hops', cols: [fromTask('chain_hop4', E, '4단계')] },
    { title: '저장 후 지연', href: '#delay-sec', cols: [fromBucket(2, '64–128'), fromBucket(4, '256–512')] },
    { title: '새 과제', href: '#suite-sec', cols: [fromTask('cyc', isSuite, '순환 사슬'), fromTask('count', isSuite, '셈'), fromTask('copy', isSuite, '베끼기'), fromTask('paren', isSuite, '괄호'), fromTask('slowp', isSuite, '느린 주기')] },
  ];
}
function drawBoard() {
  const groups = boardColumns();
  const cols = groups.flatMap((g) => g.cols);
  const table = $('#board-table');
  const avgCols = groups.find((g) => g.avg).cols;
  const avgOf = (v) => { const a = avgCols.map((c) => mean(c.values(v))); return a.some(Number.isNaN) ? NaN : mean(a); };
  let html = '<thead><tr><th class="name"></th>' + groups.map((g) => `<th colspan="${g.cols.length + (g.avg ? 1 : 0)}"><a href="${g.href}">${g.title}</a></th>`).join('') + '</tr>';
  html += '<tr><th class="name"></th>' + groups.map((g) => g.cols.map((c) => `<th>${c.label}</th>`).join('') + (g.avg ? '<th>평균</th>' : '')).join('') + '</tr></thead><tbody>';
  for (const v of FOCUS_ROSTER) {
    html += `<tr data-m="${v}"><th class="name"><span class="sw" style="background:${colorOf(v)}"></span>${MODELS[v].name}</th>`;
    let i = 0;
    for (const g of groups) {
      for (const c of g.cols) {
        const a = c.values(v), val = mean(a);
        if (Number.isNaN(val)) html += '<td class="none">–</td>';
        else {
          const top = Math.max(...FOCUS_ROSTER.map((w) => mean(c.values(w))).filter((q) => !Number.isNaN(q)));
          const near = c.rule != null && Math.abs(val - c.rule) <= 0.02;
          const style = `background:color-mix(in oklab, var(--seq) ${Math.round(val * 100)}%, var(--page))`;
          // 모두 우연 근처인 열에는 1등을 표시하지 않는다
          const best = val >= top - 0.005 && top >= c.chance + 0.05;
          const flip = val >= 0.8 ? 'flip' : val >= 0.7 ? 'flip-dark' : '';
          html += `<td data-c="${i}" class="${best ? 'best ' : ''}${near ? 'near ' : ''}${flip}" style="${style}">${(val * 100).toFixed(0)}</td>`;
        }
        i++;
      }
      if (g.avg) { const q = avgOf(v); html += `<td class="avg">${Number.isNaN(q) ? '–' : (q * 100).toFixed(1)}</td>`; }
    }
    html += '</tr>';
  }
  table.innerHTML = html + '</tbody>';
  table.querySelectorAll('td[data-c]').forEach((td) => {
    const c = cols[+td.dataset.c], v = td.parentElement.dataset.m, a = c.values(v);
    td.addEventListener('mousemove', (ev) => showTip(ev, `${MODELS[v].name} · ${c.label}`, [
      ['평균', pct(mean(a)), colorOf(v)], ['난수별', a.map((q) => p1(q)).join(' · ')],
      ...(c.oracle != null ? [['이론적 최대', pct(c.oracle)]] : []),
      ...(c.rule != null ? [[`규칙: ${c.ruleName}`, pct(c.rule)]] : []),
    ]));
    td.addEventListener('mouseleave', hideTip);
  });
}

// ---------------------------------------------------------------- 직접 풀기: 실제 시험 문제와 모델의 실제 답
const VALUES = '12345678', KEYS = 'ABCDEFGH', ENTS = '가나다라마바사아';
const valOf = (t) => VALUES[t], keyOf = (t) => KEYS[t - 8], entOf = (t) => ENTS[t - 16];
const tokHtml = (s, cls = '') => `<span class="tok ${cls}">${s}</span>`;
const GAMES = {
  kv: {
    name: '키-값 기억', ask: (q) => `${keyOf(q.prompt[q.prompt.length - 1])}에 지금 들어 있는 값은?`,
    options: [...Array(8).keys()], glyph: (t) => (t >= 0 && t < 8 ? valOf(t) : '다른 기호'),
    ops: (q) => { const p = q.prompt, o = []; for (let i = 0; i + 2 < p.length; i += 3) o.push({ store: p[i] === 24, k: p[i + 1], v: p[i + 2], i }); return o; },
    tape(q, done) {
      const ops = this.ops(q), k = q.prompt[q.prompt.length - 1];
      const last = ops.filter((o) => o.k === k).pop();
      return ops.map((o) => tokHtml(o.store ? `${keyOf(o.k)}=${valOf(o.v)}` : `${keyOf(o.k)}? ${valOf(o.v)}`, `${o.store ? '' : 'old'} ${done && o === last ? 'hit' : ''}`)).join('');
    },
    why(q) { return `${keyOf(q.prompt[q.prompt.length - 1])}에 마지막으로 나온 값은 ${valOf(q.answer)}이다.`; },
    flash: true,
  },
  chain: {
    name: '관계 사슬', ask: (q) => `${entOf(q.prompt[q.prompt.length - 1])}에서 화살표를 끝까지 따라가면?`,
    options: [...Array(8).keys()].map((i) => i + 16), glyph: (t) => (t >= 16 && t < 24 ? entOf(t) : '다른 기호'),
    tape(q) {
      const p = q.prompt; let h = '';
      for (let i = 1; i < p.length - 2;) {
        if (p[i] === 25) { h += tokHtml(`${entOf(p[i + 1])} 끝? ${entOf(p[i + 2])}`, 'old'); i += 3; } else { h += tokHtml(`${entOf(p[i])}→${entOf(p[i + 2])}`); i += 3; }
      }
      return h;
    },
    why(q) {
      const p = q.prompt, nx = {};
      for (let i = 1; i + 2 < p.length; i++) if (p[i] === 26) nx[p[i - 1]] = p[i + 1];
      const path = [p[p.length - 1]];
      while (nx[path[path.length - 1]] != null) path.push(nx[path[path.length - 1]]);
      return `${path.map(entOf).join(' → ')}`;
    },
  },
  paren: {
    name: '괄호 닫기', ask: () => '다음에 닫히는 괄호는?',
    options: [8, 9], glyph: (t) => ({ 8: ')', 9: ']' })[t] || '다른 기호',
    tape(q) {
      const p = q.prompt; let h = '';
      for (let i = 0; i < p.length - 1; i++) {
        if (p[i] === 29) h += tokHtml('(', 'bare');
        else if (p[i] === 30) h += tokHtml('[', 'bare');
        else if (p[i] === 25) { h += tokHtml(p[i + 1] === 8 ? ')' : ']', 'bare'); i++; } else h += tokHtml('·', 'fill');
      }
      return h + tokHtml('?', 'q');
    },
    why(q) { return `가장 안쪽에 열려 있는 괄호는 ${q.answer === 8 ? '(' : '['} 이다. 점은 뜻 없는 기호다.`; },
  },
  copy: {
    name: '베끼기 · 뒤집기', ask: () => '다음 기호는?',
    options: [...Array(16).keys()], glyph: (t) => (t >= 0 && t < 8 ? valOf(t) : t >= 8 && t < 16 ? keyOf(t) : '다른 기호'),
    tape(q) {
      const p = q.prompt, g = (t) => this.glyph(t); let h = '';
      for (let i = 1; i < p.length; i++) h += p[i] === 29 ? tokHtml('그대로', 'tag') : p[i] === 30 ? tokHtml('거꾸로', 'tag') : tokHtml(g(p[i]));
      return h + tokHtml('?', 'q');
    },
    why(q) { const j = q.prompt.findIndex((t) => t === 29 || t === 30); return `${q.prompt[j] === 30 ? '원문을 거꾸로' : '원문을 그대로'} 내는 중이고, ${q.prompt.length - j}번째 기호는 ${this.glyph(q.answer)}이다.`; },
  },
};
const play = { task: 'kv', memory: true, st: {}, timer: null };
function playState(task) {
  return play.st[task] || (play.st[task] = { i: 0, picked: null, me: 0, done: 0, models: {} });
}
function stopFlash() { clearTimeout(play.timer); play.timer = null; }
function drawPlay() {
  picker($('#play-picker'), '문제', Object.keys(GAMES).map((k) => [k, GAMES[k].name]), play.task, (k) => { play.task = k; stopFlash(); drawPlay(); });
  stopFlash();
  const stage = $('#stage'), g = GAMES[play.task], data = D.samples.tasks[play.task], st = playState(play.task);
  if (st.i >= data.questions.length) { drawPlayEnd(stage, g, data, st); return; }
  const q = data.questions[st.i], answered = st.picked != null;
  const flash = g.flash && play.memory && !answered;
  stage.innerHTML = `<div class="stage-top"><span>${st.i + 1} / ${data.questions.length}번 문제</span><span>${answered ? '' : g.flash ? `<button type="button" class="txt-btn" id="mode">${play.memory ? '보면서 풀기' : '기억하며 풀기'}</button>` : ''}</span></div>
    <div class="tape" id="tape">${flash ? '' : g.tape(q, answered)}</div>
    <p class="ask" id="ask"${flash ? ' hidden' : ''}>${g.ask(q)}</p>
    <div class="opts" id="opts"${flash ? ' hidden' : ''}></div><div id="after"></div>`;
  const mode = $('#mode');
  if (mode) mode.addEventListener('click', () => { play.memory = !play.memory; drawPlay(); });
  const opts = $('#opts');
  for (const o of g.options) {
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = g.glyph(o);
    if (answered) { b.disabled = true; if (o === q.answer) b.className = 'right'; else if (o === st.picked) b.className = 'wrong'; }
    else b.addEventListener('click', () => answer(q, o));
    opts.appendChild(b);
  }
  if (flash) runFlash(g, q);
  if (answered) drawAfter(g, q, st, data);
}
function runFlash(g, q) {
  const ops = g.ops(q), tape = $('#tape');
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  let i = 0;
  const step = () => {
    if (i >= ops.length) {
      tape.innerHTML = '<div class="flash" style="width:100%"><small>이제 가렸다. 기억나는가?</small></div>';
      $('#ask').hidden = false; $('#opts').hidden = false;
      return;
    }
    const o = ops[i];
    tape.innerHTML = `<div class="flash" style="width:100%">${tokHtml(o.store ? `${keyOf(o.k)}=${valOf(o.v)}` : `${keyOf(o.k)}? ${valOf(o.v)}`, o.store ? '' : 'old')}<small>${i + 1} / ${ops.length}</small></div>`;
    i++;
    play.timer = setTimeout(step, reduce ? 1400 : 850);
  };
  step();
}
function answer(q, o) {
  const st = playState(play.task);
  st.picked = o; st.done++;
  if (o === q.answer) st.me++;
  for (const [v, p] of Object.entries(q.preds)) st.models[v] = (st.models[v] || 0) + p.filter((x) => x === q.answer).length;
  drawPlay();
}
function playFilter(task) { return D.samples.tasks[task].group === 'E' ? COND.E.f : isSuite; }
function answerRows(q, st, task) {
  const f = playFilter(task), g = GAMES[task];
  const vs = FOCUS_ROSTER.filter((v) => (q ? q.preds[v] : st.models[v] != null));
  return vs.map((v) => {
    const c = colorOf(v), test = mean(vals(v, f, task)), so = (st.models[v] || 0) / (st.done * 3);
    let dots = '', said = '';
    if (q) {
      dots = q.preds[v].map((x) => `<i class="${x === q.answer ? 'y' : ''}"></i>`).join('');
      const tally = {}; q.preds[v].forEach((x) => { tally[x] = (tally[x] || 0) + 1; });
      said = g.glyph(+Object.keys(tally).sort((a, b) => tally[b] - tally[a])[0]);
    }
    return `<div class="ans" data-m="${v}" style="--c:${c}"><span class="who"><span class="sw" style="background:${c}"></span>${MODELS[v].name}</span>
      <span class="dots" aria-label="난수 3개 중 ${q ? q.preds[v].filter((x) => x === q.answer).length : ''}개 정답">${dots}</span><span class="said">${said}</span>
      <span class="so" title="지금까지 ${(so * 100).toFixed(0)}%"><b style="width:${so * 100}%"></b></span><span class="test">${pct(test, 0)}</span></div>`;
  }).join('');
}
const ansHead = (one = true) => `<div class="ans head"><span></span><span>${one ? '난수 3개' : ''}</span><span>${one ? '답' : ''}</span><span>${one ? '지금까지 푼 문제' : '이 8문제 정답률'}</span><span style="text-align:right">전체 시험</span></div>`;
function drawAfter(g, q, st, data) {
  const ok = st.picked === q.answer, n = Object.keys(q.preds).length;
  const allRight = Object.values(q.preds).filter((p) => p.every((x) => x === q.answer)).length;
  const last = st.i + 1 >= data.questions.length;
  $('#after').innerHTML = `<p class="verdict">${ok ? '맞았다.' : `틀렸다. 답은 ${g.glyph(q.answer)}.`} ${allRight ? `모델 ${n}종 중 ${allRight}종은 난수 셋 다 맞혔다.` : '난수 셋 다 맞힌 모델은 없다.'}</p><p class="why">${g.why(q)}</p>
    <div class="answers">${ansHead()}${answerRows(q, st, play.task)}</div>
    <div class="foot-row"><button type="button" class="txt-btn primary" id="next">${last ? '결과 보기' : '다음 문제'}</button><span class="note" style="margin:0">내 점수 ${st.me} / ${st.done}</span></div>`;
  $('#next').addEventListener('click', () => { st.i++; st.picked = null; drawPlay(); });
}
function drawPlayEnd(stage, g, data, st) {
  const best = FOCUS_ROSTER.filter((v) => st.models[v] != null).sort((a, b) => st.models[b] - st.models[a])[0];
  stage.innerHTML = `<div class="tile" style="border:0;padding:0"><span class="big">${st.me} / ${st.done}</span><span class="what">${g.name} 내 점수</span><span class="sub">같은 문제에서 ${MODELS[best].name}는 ${((st.models[best] / (st.done * 3)) * 100).toFixed(0)}%를 맞혔다</span></div>
    <div class="answers">${ansHead(false)}${answerRows(null, st, play.task)}</div>
    <div class="foot-row"><button type="button" class="txt-btn" id="again">다시 풀기</button>${Object.keys(GAMES).filter((k) => k !== play.task).map((k) => `<button type="button" class="txt-btn" data-go="${k}">${GAMES[k].name} 풀기</button>`).join('')}</div>`;
  $('#again').addEventListener('click', () => { delete play.st[play.task]; drawPlay(); });
  stage.querySelectorAll('[data-go]').forEach((b) => b.addEventListener('click', () => { play.task = b.dataset.go; drawPlay(); }));
}

// ---------------------------------------------------------------- 구조: 기호를 넣으면 무엇이 바뀌나 (설명용 그림)
const sch = { t: 0, timer: null };
function drawSchematicsAt(t) {
  $('#sch-label').textContent = `기호 ${t}`;
  $('#sch-t').value = t;
  // Folio: 기호마다 한 칸에 쓰고 두 칸을 읽는다 (칸 번호는 설명용 난수)
  const f = $('#folio-svg');
  f.innerHTML = '';
  el('rect', { x: 6, y: 30, width: 62, height: 32, rx: 3, fill: 'none', stroke: 'var(--folio)', 'stroke-width': 1.5 }, f);
  text(f, 37, 50, '제어', { 'text-anchor': 'middle', class: 't-ink2' });
  const w = (t * 13 + 5) % 32, r1 = (t * 7 + 21) % 32, r2 = (t * 11 + 2) % 32;
  const slot = (i) => [140 + (i % 8) * 19, 18 + Math.floor(i / 8) * 15];
  for (let i = 0; i < 32; i++) {
    const [x, y] = slot(i), written = i === w, read = i === r1 || i === r2;
    el('rect', { x, y, width: 15, height: 11, rx: 1.5, style: `fill:${written ? 'var(--folio)' : read ? 'var(--ink-2)' : 'none'};stroke:${read ? 'var(--ink-2)' : 'var(--folio)'};stroke-width:1` }, f);
  }
  const [wx, wy] = slot(w), [rx, ry] = slot(r1);
  el('path', { d: `M68 40 C 95 ${wy}, 110 ${wy + 5}, ${wx} ${wy + 5}`, fill: 'none', style: 'stroke:var(--folio);stroke-width:1.2' }, f);
  el('path', { d: `M${rx} ${ry + 5} C 110 ${ry + 5}, 95 70, 68 54`, fill: 'none', style: 'stroke:var(--ink-2);stroke-width:1' }, f);
  text(f, 76, 16, '파랑: 쓰기', { class: 't-num' });
  text(f, 76, 88, '회색: 읽기 2번', { class: 't-num' });
  // Strata: 지금 고쳐지는 층의 칸만 진하게
  const s = $('#strata-svg');
  s.innerHTML = '';
  const u = 3.8, tt = t % 64;
  [64, 16, 4, 1].forEach((p, row) => {
    const y = 8 + row * 20, fresh = tt % p === 0;
    text(s, 0, y + 10, `${p}마다`, { class: fresh ? 't-num t-ink' : 't-num' });
    for (let c = 0; c < 64; c += p) {
      const on = tt >= c && tt < c + p;
      el('rect', { x: 50 + c * u, y, width: Math.max(1.6, p * u - 1.6), height: 13, rx: 1.5, style: `fill:var(--strata);fill-opacity:${on ? (fresh ? 1 : 0.55) : 0.14}` }, s);
    }
  });
  // Lattice: 입력 칸에서 메시지가 세 번 퍼진다. 격자의 양끝은 이어져 있다.
  const l = $('#lattice-svg');
  l.innerHTML = '';
  const cell = 17, ox = 92, k = t % 4;
  const torus = (a) => Math.min(a, 4 - a);
  for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) {
    const reached = torus(r) + torus(c) <= k, never = torus(r) + torus(c) > 3;
    el('rect', { x: ox + c * (cell + 4), y: 4 + r * (cell + 4), width: cell, height: cell, rx: 2, style: `fill:var(--lattice);fill-opacity:${reached ? (r === 0 && c === 0 ? 1 : 0.55) : 0};stroke:${never ? 'var(--bad)' : 'var(--lattice)'};stroke-width:${never ? 1.6 : 1.2}` }, l);
  }
  text(l, ox - 10, 16, '입력', { 'text-anchor': 'end', class: 't-num' });
  text(l, ox + 94, 26, `주고받기 ${k}번`, { class: 't-num t-ink2' });
  text(l, ox + 94, 58, '빨강: 세 번으론', { class: 't-num' });
  text(l, ox + 94, 72, '못 닿는 칸', { class: 't-num' });
}
function setupSchematics() {
  const btn = $('#sch-play'), range = $('#sch-t');
  const stop = () => { clearInterval(sch.timer); sch.timer = null; btn.textContent = '▶ 재생'; btn.setAttribute('aria-pressed', 'false'); };
  const start = () => {
    sch.timer = setInterval(() => { sch.t = (sch.t + 1) % 64; drawSchematicsAt(sch.t); }, 450);
    btn.textContent = '❚❚ 멈춤'; btn.setAttribute('aria-pressed', 'true');
  };
  btn.addEventListener('click', () => (sch.timer ? stop() : start()));
  range.addEventListener('input', () => { stop(); sch.t = +range.value; drawSchematicsAt(sch.t); });
  sch.touched = false;
  btn.addEventListener('click', () => { sch.touched = true; }, { once: true });
  range.addEventListener('input', () => { sch.touched = true; }, { once: true });
  // 화면에 들어오면 저절로 움직인다 (움직임 줄이기 설정이면 멈춰 둔다)
  if (!matchMedia('(prefers-reduced-motion: reduce)').matches && 'IntersectionObserver' in window) {
    new IntersectionObserver((es) => {
      for (const e of es) {
        if (e.isIntersecting && !sch.timer && !sch.touched) start();
        else if (!e.isIntersecting && sch.timer && !sch.touched) stop();
      }
    }).observe($('#build'));
  }
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
  $('#acc-cap').innerHTML = `작은 점이 난수 하나(${n}개), 큰 점이 평균이다.`
    + (cond === 'D' ? ' 과제 하나씩 학습한 모델은 그 문제와 긴 변형만 시험했다.' : '')
    + (metric === 'bits' ? ' 손실은 0에 가까울수록 확신을 갖고 맞힌 것이다.' : '');
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
  $('#curve-cap').innerHTML = `800 → 3000걸음 평균 상승(시험 정확도, 난수 ${nE}개): `
    + big.map(([v, g]) => `${MODELS[v].name} <b>${g >= 0 ? '+' : '−'}${Math.abs(g * 100).toFixed(1)}%p</b>`).join(' · ')
    + '. 곡선이 끝에서 평평한 건 학습률이 0으로 줄기 때문이라, 다 배웠다는 뜻은 아니다.';
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
    ['함께 학습 · 800걸음', COND.A.f, '난수 3개 평균'],
    ['관계 사슬만 · 400걸음', (r) => COND.D.f(r) && r.tasks[0] === 'chain', '난수 3개 평균'],
    ['함께 학습 · 3000걸음', COND.E.f, `난수 ${new Set(D.runs.filter(COND.E.f).map((r) => r.seed)).size}개 평균`],
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
  $('#hop-cap').innerHTML = `3000걸음, 학습보다 한 단계 긴 4단계: `
    + MAIN.map((v) => `${MODELS[v].name} <b>${pct(e(v, 'chain_hop4'))}</b>`).join(' · ')
    + '. 5단계 문제는 짧은 사슬이 하나뿐이라 찍기가 쉽다.';
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
  picker($('#delay-picker'), '학습 채점', have.map((t) => [t, t === 'kvd' ? '숨긴 키만 채점' : '모든 질의 채점']), delayTrain,
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
  refKey(key, [['line', '청크 256'], ['thin', '청크 64'], ['dash', '최근 저장값 규칙']]);
  const series = combos.map(({ v, c, rs }) => ({
    key: v, thin: c === 64, points: delayCurve(rs).map(([i, m]) => [i, m]),
    label: c === 256 || v === 'transformer' ? `${MODELS[v].name} ${c}` : null, tipLabel: `${MODELS[v].name} · 청크 ${c} (난수 ${rs.length})`,
  }));
  const base = D.delay_baseline ? BUCKETS.map((_, i) => bucketAcc(D.delay_baseline, i)).filter((v) => !Number.isNaN(v)) : [];
  const box = $('#delay-chart');
  lineChart(box, {
    series, xDomain: [-0.3, BUCKETS.length - 0.7], xTicks: BUCKETS.map((_, i) => i), xFmt: (i) => BUCKETS[i], height: 280,
    refs: [{ v: mean(base), cls: 'ref-rule' }, { v: 0.125, cls: 'grid' }],
    vlines: [{ x: 1.5, label: '64기호' }, { x: 3.5, label: '학습 최대 지연' }],
  });
  const at = (v, c, b) => { const cc = combos.find((x) => x.v === v && x.c === c); if (!cc) return '–'; const p = delayCurve(cc.rs).find((q) => BUCKETS[q[0]] === b); return p ? pct(p[1]) : '–'; };
  const head = delayTrain === 'kvd'
    ? `청크 256이면 창 밖도 맞힌다. 256–512기호에서 GRU는 <b>${at('gru', 64, '256-512')} → ${at('gru', 256, '256-512')}</b>, Folio는 <b>${at('folio', 64, '256-512')} → ${at('folio', 256, '256-512')}</b>로 올랐다. Transformer(창 64)는 숨긴 키만 채점하면 창 안도 잘 못 배운다.`
    : `방해 구간까지 채점하니 Transformer는 창 안(32–64기호)이 <b>${at('transformer', 64, '32-64')}</b>까지 올랐다. GRU와 Folio는 최근 키만 좇게 되어 같은 구간이 <b>${at('gru', 256, '32-64')}</b>, <b>${at('folio', 256, '32-64')}</b>로 떨어졌다.`;
  $('#delay-cap').innerHTML = `${head} 시험은 어느 쪽이든 숨긴 키만 묻는다. 2000걸음, 난수 3개 평균. 우연은 12.5%.`;
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
  refKey($('#suite-key'), [['dash', '단순 규칙'], ['thin', 'Strata 층 없앤 대조군 (느린 주기만)']]);
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
  $('#suite-cap').innerHTML = `함께 학습 · 3000걸음 · 난수 ${new Set(H.map((r) => r.seed)).size}개 평균. 가는 가로선이 우연 수준이다. 그래프에 마우스를 올리면 값이 나온다.`;
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
    caps.push(`${MODELS[v2].name}는 ${ok.length}개 비교에서 <b>${up}개 오르고 ${down}개 내렸다</b>(2%p 넘게 바뀐 것만, 평균 ${(mean(ok.map((r) => r.b - r.a)) * 100).toFixed(1)}%p)`);
  }
  $('#v2-cap').innerHTML = caps.join('. ') + '. 난수 3개 평균. 그래도 둘 다 GRU 아래다.';
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
    dotRows(panel(box, MODELS[base].name, '800걸음'), { rows, domain: [0, 1], refs });
    for (const r of rows.slice(1)) caps.push(`${nameOf(r.key)} ${p1(bm)} → <b>${p1(mean(r.values))}</b>`);
  }
  $('#abl-cap').innerHTML = `<b>${TASK[ablTask].name}</b>: ` + caps.join(' · ') + ' (%, 난수 3개 평균). 다중 주기 대조군은 파라미터가 같고 연산이 더 많다.';
}

// ---------------------------------------------------------------- 7. 비용
function drawCosts() {
  const M = D.measure;
  if (!M) { $('#cost-cap').textContent = '자원 측정 결과가 없다 (py main.py가 잰다).'; return; }
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
      const mk = MODELS[r.variant].base || r.variant, g = el('g', { 'data-m': mk }, svg);
      el('line', { x1: x(ticks[0]), x2: x(v), y1: cy, y2: cy, style: `stroke:${c};stroke-width:2;stroke-opacity:0.35` }, g);
      el('circle', { cx: x(v), cy, r: 5, style: ring ? `fill:var(--page);stroke:${c};stroke-width:2` : `fill:${c};stroke:var(--page);stroke-width:2` }, g);
      text(g, x(v) + 9, cy + 4, fmt(v), { class: 't-num t-ink2' });
    });
  };
  logRows(panel(multi, '학습 한 걸음', '짧을수록 좋다'), (r) => r.train_sec_per_step * 1000,
    (v) => (v >= 1000 ? `${+(v / 1000).toFixed(1)}초` : `${Math.round(v)}ms`), [1, 10, 100, 1000], 3000);
  logRows(panel(multi, '학습 피크 메모리', 'GPU'), (r) => r.train_peak_bytes / 2 ** 20,
    (v) => (v >= 1000 ? `${+(v / 1024).toFixed(1)}GiB` : `${Math.round(v)}MiB`), [10, 100, 1000], 3000);
  const tf = rows.find((r) => r.variant === 'transformer');
  const ratios = CANDS.map((v) => rows.find((r) => r.variant === v)).filter(Boolean).map((r) => r.train_sec_per_step / tf.train_sec_per_step);
  $('#cost-cap').innerHTML = `후보 세 개의 학습 한 걸음은 Transformer의 <b>${Math.min(...ratios).toFixed(0)}–${Math.max(...ratios).toFixed(0)}배</b>다. 빈 원은 대조군이다. 가로축은 로그 눈금이다.`;
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
  if (!C) { $('#check-cap').textContent = '검증 결과가 없다 (py main.py가 검사한다).'; return; }
  const checks = [...new Set(C.rows.filter((r) => r.variant).map((r) => r.check))];
  const variants = [...new Set(C.rows.filter((r) => r.variant).map((r) => r.variant))];
  const NAMES = { 'forward = step 누적': '한 번에 = 한 개씩', '인과성': '인과성', '상태 누출 없음': '누출 없음', '청크 분할 일치': '청크 분할', '유한 손실·gradient': '유한 손실·기울기', '작은 결정적 문제 학습': '작은 문제 학습' };
  const cell = (r) => (r ? `<td title="${r.msg.replace(/"/g, '&quot;')}">${r.ok ? '<span class="ok">✓ 통과</span>' : '<span class="fail">✗ 실패</span>'}</td>` : '<td class="dim">–</td>');
  $('#check-table').innerHTML = '<thead><tr><th class="l">모델</th>' + checks.map((c) => `<th>${NAMES[c] || c}</th>`).join('') + '</tr></thead><tbody>'
    + variants.map((v) => `<tr><td class="l">${MODELS[v] ? nameOf(v) : v}</td>` + checks.map((c) => cell(C.rows.find((r) => r.variant === v && r.check === c))).join('') + '</tr>').join('')
    + '</tbody>';
  const gen = C.rows.find((r) => !r.variant);
  const failed = C.rows.filter((r) => !r.ok).length;
  $('#check-tiles').innerHTML = `<div class="tile"><span class="big">${C.rows.length - failed} / ${C.rows.length}</span><span class="what">검사 통과</span><span class="sub">64비트 정밀도로 모델별 같은 검사</span></div>
    <div class="tile"><span class="big">${variants.length}개</span><span class="what">모델과 대조군</span><span class="sub">한 번에 넣든 나눠 넣든 같은 답이 나오는지, 미래 기호가 새지 않는지 등</span></div>`;
  $('#check-cap').innerHTML = `${C.checked_at} · ${C.device} · ${C.dtype}. 과제 생성기: ${gen ? (gen.ok ? '모든 질의의 답을 따로 다시 계산해 일치 확인' : '실패 — ' + gen.msg) : '–'}. 칸에 마우스를 올리면 잰 값이 보인다.`;
}

// ---------------------------------------------------------------- 전체
function drawAll() {
  drawChips();
  drawTiles();
  drawBoard();
  drawSchematicsAt(sch.t);
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
    st.textContent = `결과를 못 불러왔다: ${err.message}.` + (window.RESULTS_URL ? '' : ' py main.py로 서버를 띄웠는지 확인해 보자.');
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
  st.remove();
  $('#foot').innerHTML = `자료 생성 ${D.generated_at} · 실행 ${D.runs.length}개 · 계획 ` + Object.entries(D.plan).map(([g, p]) => `${g} ${p.done}/${p.total}`).join(' · ')
    + (D.measure ? ` · 자원 측정 ${D.measure.measured_at}` : '') + '. 숫자는 전부 실행 결과 파일에서 화면이 바로 계산한다.';
  drawAll();
  if (D.samples) drawPlay(); else $('#play').hidden = true;
  setupSchematics();
  $('#suite-more').addEventListener('toggle', drawSuite);
  let w = innerWidth;
  addEventListener('resize', () => { if (Math.abs(innerWidth - w) > 30) { w = innerWidth; drawAll(); } });
}

main();
