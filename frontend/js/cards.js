/**
 * cards.js — Game card rendering, skeleton states, error/empty states.
 *
 * Exports:
 *   renderSkeletons(n)         → show N shimmer skeleton cards
 *   renderCards(items, mode)   → render game cards in the results grid
 *   renderError(msg)           → show error state
 *   renderEmpty()              → show empty state
 *   renderLLMSummary(text)     → show AI summary blockquote
 *   clearResults()             → wipe all results
 */

const grid        = () => document.getElementById('results-grid');
const llmBox      = () => document.getElementById('llm-summary-box');
const metaBox     = () => document.getElementById('results-meta');

/* ── Helpers ──────────────────────────────────────────────────── */

function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

/**
 * Map a raw logit/score to a 0–1 display percentage.
 * CF scores are logits (can be any range); we normalise across the batch.
 * Semantic scores are cosine similarities in [−1, 1].
 */
function normaliseScores(items) {
  if (!items.length) return items;
  const scores = items.map(it => it.score ?? it.similarity_score ?? 0);
  const min = Math.min(...scores);
  const max = Math.max(...scores);
  const range = max - min || 1;
  return items.map((it, i) => ({
    ...it,
    _normScore: (scores[i] - min) / range,
  }));
}

function sourceBadgeClass(source) {
  if (source === 'collaborative_filtering') return 'source-badge--cf';
  if (source === 'cold_start')              return 'source-badge--cold';
  return 'source-badge--chat';
}

function sourceBadgeLabel(source) {
  if (source === 'collaborative_filtering') return 'Collaborative Filtering';
  if (source === 'cold_start')              return 'Semantic Search';
  return 'Conversational';
}

/* ── Public API ───────────────────────────────────────────────── */

function clearResults() {
  grid().innerHTML = '';
  llmBox().classList.add('hidden');
  llmBox().innerHTML = '';
  metaBox().classList.add('hidden');
  metaBox().innerHTML = '';
}

function renderSkeletons(n = 6) {
  clearResults();
  grid().innerHTML = Array.from({ length: n }, () => `
    <div class="skeleton-card">
      <div class="skeleton-line skeleton-line--sm"></div>
      <div class="skeleton-line skeleton-line--xl" style="margin-bottom:6px"></div>
      <div class="skeleton-line skeleton-line--md" style="margin-bottom:18px"></div>
      <div class="skeleton-line skeleton-line--lg skeleton-line--sm"></div>
      <div class="skeleton-line skeleton-line--lg" style="margin-top:20px;height:3px"></div>
    </div>
  `).join('');
}

function renderError(msg) {
  clearResults();
  grid().innerHTML = `
    <div class="state-card state-card--error">
      <div class="state-icon">⚠️</div>
      <h3 class="state-title">Something went wrong</h3>
      <p class="state-desc">${escHtml(msg)}</p>
    </div>
  `;
}

function renderEmpty() {
  clearResults();
  grid().innerHTML = `
    <div class="state-card">
      <div class="state-icon">🎮</div>
      <h3 class="state-title">No recommendations found</h3>
      <p class="state-desc">Try a different user ID, add more games, or rephrase your query.</p>
    </div>
  `;
}

function renderLLMSummary(text, mode) {
  if (!text) return;
  const box = llmBox();
  box.innerHTML = `
    <div class="llm-summary-label">✨ AI Summary</div>
    ${escHtml(text)}
  `;
  box.classList.remove('hidden');
}

/**
 * renderResultsMeta — show source badge + count
 * @param {string} source  - 'collaborative_filtering' | 'cold_start' | 'conversational'
 * @param {number} count   - number of results
 * @param {string} [userId]
 */
function renderResultsMeta(source, count, userId) {
  const box = metaBox();
  const badgeClass = sourceBadgeClass(source);
  const badgeLabel = sourceBadgeLabel(source);
  const userPart = userId ? `· user <code>${escHtml(userId)}</code>` : '';
  box.innerHTML = `
    <span class="source-badge ${badgeClass}">${badgeLabel}</span>
    <span>${count} recommendations ${userPart}</span>
  `;
  box.classList.remove('hidden');
}

/**
 * renderCards — main rendering function.
 * @param {object[]} items  - recommendation objects from API
 * @param {string}   mode   - 'cf' | 'coldstart' | 'chat'
 * @param {string}  [userId]
 */
function renderCards(items, mode, userId) {
  clearResults();
  if (!items || !items.length) { renderEmpty(); return; }

  const normalised = normaliseScores(items);
  const source = items[0]?.source ?? (mode === 'cf' ? 'collaborative_filtering' : mode === 'coldstart' ? 'cold_start' : 'conversational');

  renderResultsMeta(source, items.length, userId);

  grid().innerHTML = normalised.map((item, idx) => {
    const title      = escHtml(item.title ?? item.item_id ?? `Game ${idx + 1}`);
    const explanation= escHtml(item.explanation ?? '');
    const hasExpl    = Boolean(item.explanation);
    const normPct    = Math.round((item._normScore ?? 0.5) * 100);
    const rawScore   = (item.score ?? item.similarity_score ?? 0).toFixed(3);
    const cardId     = `card-${idx}`;

    return `
      <div class="game-card" id="${cardId}" data-idx="${idx}">
        <div class="card-inner">

          <!-- Front -->
          <div class="card-front">
            <div class="card-rank">#${idx + 1} &nbsp;·&nbsp; score ${rawScore}</div>
            <div class="card-title">${title}</div>
            <div class="card-tags">${item.source ? sourceBadgeLabel(item.source) : ''}</div>

            <div class="score-bar-wrap">
              <div class="score-bar-header">
                <span class="score-label">Relevance</span>
                <span class="score-val">${normPct}%</span>
              </div>
              <div class="score-track">
                <div class="score-fill" style="width:${normPct}%"></div>
              </div>
            </div>

            <div class="card-actions">
              ${hasExpl
                ? `<button class="btn-flip" onclick="flipCard('${cardId}')">Why? 🔍</button>`
                : `<button class="btn-flip" style="opacity:0.4;cursor:default">No explanation</button>`
              }
            </div>
          </div>

          <!-- Back -->
          <div class="card-back">
            <div class="card-back-label">💡 Why this game?</div>
            <div class="card-explanation">
              ${explanation || '<em style="color:var(--text-faint)">No explanation available.</em>'}
            </div>
            <div class="card-back-actions">
              <button class="btn-unflip" onclick="flipCard('${cardId}')">← Back</button>
            </div>
          </div>

        </div>
      </div>
    `;
  }).join('');
}

/* Global flip handler (called from inline onclick) */
function flipCard(cardId) {
  const card = document.getElementById(cardId);
  if (card) card.classList.toggle('flipped');
}
