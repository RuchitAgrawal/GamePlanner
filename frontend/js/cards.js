/**
 * cards.js — Editorial game card rendering (no flip).
 *
 * Cards show title + explanation directly — no hidden back side.
 * Explanations are the primary content, not a secondary reveal.
 */

const grid    = () => document.getElementById('results-grid');
const llmBox  = () => document.getElementById('llm-summary-box');
const metaBox = () => document.getElementById('results-meta');
const resultArea = () => document.getElementById('results-area');

/* ── Helpers ──────────────────────────────────────────────────── */
function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function normaliseScores(items) {
  if (!items.length) return items;
  const scores = items.map(it => it.score ?? it.similarity_score ?? 0);
  const min = Math.min(...scores);
  const max = Math.max(...scores);
  const range = max - min || 1;
  return items.map((it, i) => ({ ...it, _pct: Math.round(((scores[i] - min) / range) * 100) }));
}

function badgeClass(mode) {
  if (mode === 'cf')       return 'badge--cf';
  if (mode === 'coldstart') return 'badge--cold';
  return 'badge--chat';
}
function badgeLabel(mode) {
  if (mode === 'cf')        return 'Collaborative Filtering';
  if (mode === 'coldstart') return 'Taste Profile';
  return 'Conversational';
}

/* ── Public API ───────────────────────────────────────────────── */

function clearResults() {
  grid().innerHTML = '';
  llmBox().classList.add('hidden');
  llmBox().innerHTML = '';
  metaBox().classList.add('hidden');
  metaBox().innerHTML = '';
  resultArea()?.classList.add('hidden');
}

function renderSkeletons(n = 6) {
  clearResults();
  resultArea()?.classList.remove('hidden');
  // Update header
  document.getElementById('results-title').textContent = 'Finding recommendations…';
  document.getElementById('results-badge').textContent = '';
  grid().innerHTML = Array.from({ length: n }, () => `
    <div class="skeleton-card">
      <div class="skeleton-line skeleton-line--sm"></div>
      <div class="skeleton-line skeleton-line--md" style="height:20px;margin-bottom:4px"></div>
      <div class="skeleton-line skeleton-line--lg" style="margin-bottom:2px"></div>
      <div class="skeleton-line skeleton-line--lg skeleton-line--xl"></div>
      <div class="skeleton-line skeleton-line--sm" style="margin-top:16px;height:2px"></div>
    </div>
  `).join('');
}

function renderError(msg) {
  clearResults();
  resultArea()?.classList.remove('hidden');
  document.getElementById('results-title').textContent = 'Error';
  document.getElementById('results-badge').textContent = '';
  grid().innerHTML = `
    <div class="state-card state-card--error">
      <div class="state-icon">⚠</div>
      <h3 class="state-title">Something went wrong</h3>
      <p class="state-desc">${escHtml(msg)}</p>
    </div>`;
}

function renderEmpty() {
  clearResults();
  resultArea()?.classList.remove('hidden');
  document.getElementById('results-title').textContent = 'No results';
  grid().innerHTML = `
    <div class="state-card">
      <div class="state-icon">🎮</div>
      <h3 class="state-title">No recommendations found</h3>
      <p class="state-desc">Try a different profile, add more games, or rephrase your query.</p>
    </div>`;
}

function renderLLMSummary(text) {
  if (!text) return;
  const box = llmBox();
  box.innerHTML = `<span class="llm-label">AI Summary</span>${escHtml(text)}`;
  box.classList.remove('hidden');
}

/**
 * renderCards — main editorial card render.
 * @param {object[]} items   - recommendation objects from API
 * @param {string}   mode    - 'cf' | 'coldstart' | 'chat'
 * @param {string}  [label]  - optional header label (persona name, etc.)
 */
function renderCards(items, mode, label) {
  clearResults();
  if (!items?.length) { renderEmpty(); return; }

  resultArea()?.classList.remove('hidden');

  // Update header
  const title = label ? `For ${label}` : 'Recommendations';
  document.getElementById('results-title').textContent = title;
  const badge = document.getElementById('results-badge');
  badge.textContent = badgeLabel(mode);
  badge.className = `results-badge ${badgeClass(mode)}`;

  // Normalise scores for the bar width
  const normalised = normaliseScores(items);

  grid().innerHTML = normalised.map((item, idx) => {
    const title      = escHtml(item.title ?? item.item_id ?? `Game ${idx + 1}`);
    const explanation= item.explanation
      ? escHtml(item.explanation)
      : '<em style="color:var(--text-faint)">No explanation available.</em>';
    const rawScore   = (item.score ?? item.similarity_score ?? 0).toFixed(3);
    const pct        = item._pct ?? 50;

    return `
      <div class="game-card" id="card-${idx}">
        <div class="card-rank"># ${idx + 1}</div>
        <div class="card-title">${title}</div>
        <div class="card-explanation">${explanation}</div>
        <div class="card-score-row">
          <span class="card-score-label">Relevance</span>
          <div class="card-score-bar">
            <div class="card-score-fill" style="width:${pct}%"></div>
          </div>
          <span class="card-score-val">${pct}%</span>
        </div>
      </div>
    `;
  }).join('');

  // Scroll to results
  resultArea()?.scrollIntoView({ behavior: 'smooth', block: 'start' });
}
