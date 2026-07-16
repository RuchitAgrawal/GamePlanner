/**
 * app.js — Orchestration: persona system, mode switching, all 3 discovery paths.
 *
 * Discovery paths:
 *   A) Personas  → click a persona card → cold-start API with seed games
 *   B) Build     → tag-input → cold-start API
 *   C) Describe  → textarea → chat API
 *   (Research)   → raw user ID → CF recommend/explain API
 */

/* ── Persona definitions ─────────────────────────────────────── */
const PERSONAS = {
  fps: {
    name:  'The FPS Veteran',
    seeds: ['Counter-Strike Global Offensive', 'Team Fortress 2', 'Left 4 Dead 2'],
  },
  indie: {
    name:  'The Indie Explorer',
    seeds: ['LIMBO', 'Psychonauts', 'Hollow Knight'],
  },
  rpg: {
    name:  'The RPG Completionist',
    seeds: ['BioShock', 'Fallout 3', 'The Elder Scrolls V Skyrim'],
  },
  casual: {
    name:  'The Weekend Casual',
    seeds: ['Rocket League', 'Portal 2', "Garry's Mod"],
  },
  survival: {
    name:  'The Survival Builder',
    seeds: ["Garry's Mod", 'Terraria', "Don't Starve"],
  },
  atmos: {
    name:  'The Atmosphere Hunter',
    seeds: ['Amnesia The Dark Descent', 'LIMBO', 'BioShock Infinite'],
  },
};

/* ── Navbar scroll ───────────────────────────────────────────── */
window.addEventListener('scroll', () => {
  const nav = document.getElementById('navbar');
  if (nav) nav.classList.toggle('scrolled', window.scrollY > 40);
}, { passive: true });


/* ── Live metrics (hero) ─────────────────────────────────────── */
async function loadMetrics() {
  try {
    const data = await api.metrics();
    const best = data.best_model ?? {};
    const set = (id, val) => {
      const el = document.getElementById(id);
      if (el && val !== undefined) el.textContent = val;
    };
    const hr10 = best.hr_at_10 != null ? (best.hr_at_10 * 100).toFixed(2) + '%' : null;
    const nd10 = best.ndcg_at_10 != null ? (best.ndcg_at_10 * 100).toFixed(2) + '%' : null;
    if (hr10) set('mv-hr10', hr10);
    if (nd10) set('mv-ndcg10', nd10);
    const modelName = best.model_name ?? best.model;
    if (modelName) set('mv-model', modelName.toUpperCase());
  } catch (_) { /* silently fail — static fallbacks in HTML */ }
}


/* ── Discovery path tab switching ───────────────────────────── */
function activatePath(mode) {
  document.querySelectorAll('.path-btn').forEach(btn => {
    const active = btn.dataset.mode === mode;
    btn.classList.toggle('active', active);
    btn.setAttribute('aria-selected', active);
  });
  document.querySelectorAll('.mode-panel').forEach(panel => {
    panel.classList.toggle('active', panel.id === `panel-${mode}`);
  });
  clearResults();
}

document.querySelectorAll('.path-btn').forEach(btn => {
  btn.addEventListener('click', () => activatePath(btn.dataset.mode));
});

// Hero CTA → scroll to discover
document.getElementById('cta-discover')?.addEventListener('click', e => {
  e.preventDefault();
  document.getElementById('discover')?.scrollIntoView({ behavior: 'smooth' });
});


/* ── Path A: Personas ────────────────────────────────────────── */
async function fetchPersona(personaKey) {
  const persona = PERSONAS[personaKey];
  if (!persona) return;

  // Visual feedback: mark card selected
  document.querySelectorAll('.persona-card').forEach(c => c.classList.remove('selected'));
  document.getElementById(`persona-${personaKey}`)?.classList.add('selected');

  // Scroll to discover section
  document.getElementById('discover')?.scrollIntoView({ behavior: 'smooth', block: 'start' });

  renderSkeletons(8);

  try {
    const data = await api.coldstart(persona.seeds, 8);
    const items = data.recommendations ?? [];

    // F-E2: show LLM summary paragraph above results
    if (data.llm_summary) {
      renderLLMSummary(data.llm_summary);
    }

    // F-E1/E2: show which seeds were matched in catalog
    const matchedSeeds = data.matched_seeds ?? [];
    const metaEl = document.getElementById('results-meta');
    if (metaEl && matchedSeeds.length) {
      metaEl.innerHTML = `
        <span style="font-size:11px;letter-spacing:0.08em;text-transform:uppercase;color:var(--text-faint)">Seeds matched</span>
        ${matchedSeeds.map(s => `<span class="seed-tag" style="font-size:12px">${escHtmlLocal(s)}</span>`).join('')}
      `;
      metaEl.classList.remove('hidden');
    }

    renderCards(items, 'coldstart', persona.name);
  } catch (err) {
    renderError(err.message);
  }
}

// Wire persona card clicks
document.querySelectorAll('.persona-card').forEach(card => {
  const key = card.dataset.persona;
  card.addEventListener('click', () => fetchPersona(key));
  card.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') fetchPersona(key); });
});


/* ── Research mode toggle (demoted raw user ID) ──────────────── */
const researchToggle = document.getElementById('research-toggle');
const researchPanel  = document.getElementById('research-panel');

researchToggle?.addEventListener('click', () => {
  researchPanel.classList.toggle('open');
  researchToggle.textContent = researchPanel.classList.contains('open')
    ? 'Research mode: enter a dataset user ID directly ↑'
    : 'Research mode: enter a dataset user ID directly ↓';
});

async function fetchCF() {
  const userId = document.getElementById('user-id-input').value.trim();
  if (!userId) { document.getElementById('user-id-input').focus(); return; }
  renderSkeletons(8);
  try {
    const data = await api.recommend(userId);
    const items = data.recommendations ?? [];
    renderCards(items, 'cf', `user ${userId}`);
  } catch (err) {
    renderError(err.message);
  }
}

document.getElementById('cf-submit')?.addEventListener('click', fetchCF);
document.getElementById('user-id-input')?.addEventListener('keydown', e => {
  if (e.key === 'Enter') fetchCF();
});

// Sample ID pills
document.querySelectorAll('.pill[data-user-id]').forEach(btn => {
  btn.addEventListener('click', () => {
    document.getElementById('user-id-input').value = btn.dataset.userId;
    researchPanel.classList.add('open');
    fetchCF();
  });
});


/* ── Path B: Build Your Profile (Cold-Start tag input) ────────── */
let tags = [];

function renderTags() {
  const list = document.getElementById('tags-list');
  list.innerHTML = tags.map((tag, i) => `
    <span class="tag-chip">
      ${escHtmlLocal(tag)}
      <button class="chip-remove" data-idx="${i}" aria-label="Remove ${escHtmlLocal(tag)}">×</button>
    </span>
  `).join('');
  list.querySelectorAll('.chip-remove').forEach(btn => {
    btn.addEventListener('click', () => { tags.splice(Number(btn.dataset.idx), 1); renderTags(); });
  });
}

function addTag(value) {
  const trimmed = value.trim().replace(/,$/,'');
  if (trimmed && !tags.includes(trimmed)) { tags.push(trimmed); renderTags(); }
}

function escHtmlLocal(str) {
  return String(str).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

const tagField = document.getElementById('tag-field');
tagField?.addEventListener('keydown', e => {
  if (e.key === 'Enter' || e.key === ',') {
    e.preventDefault();
    addTag(tagField.value);
    tagField.value = '';
  } else if (e.key === 'Backspace' && !tagField.value && tags.length) {
    tags.pop(); renderTags();
  }
});
document.getElementById('tag-input-container')?.addEventListener('click', () => tagField?.focus());

async function fetchColdstart() {
  if (tagField.value.trim()) { addTag(tagField.value); tagField.value = ''; }
  if (!tags.length) { tagField?.focus(); return; }
  renderSkeletons(8);
  try {
    const data = await api.coldstart(tags, 8);

    // F-E2: LLM summary paragraph
    if (data.llm_summary) renderLLMSummary(data.llm_summary);

    // F-E1/E2: matched seeds display
    const matchedSeeds = data.matched_seeds ?? [];
    const metaEl = document.getElementById('results-meta');
    if (metaEl && matchedSeeds.length) {
      metaEl.innerHTML = `
        <span style="font-size:11px;letter-spacing:0.08em;text-transform:uppercase;color:var(--text-faint)">Seeds matched</span>
        ${matchedSeeds.map(s => `<span class="seed-tag" style="font-size:12px">${escHtmlLocal(s)}</span>`).join('')}
      `;
      metaEl.classList.remove('hidden');
    }

    renderCards(data.recommendations ?? [], 'coldstart', 'your profile');
  } catch (err) {
    renderError(err.message);
  }
}

document.getElementById('coldstart-submit')?.addEventListener('click', fetchColdstart);

document.querySelectorAll('.sample-tags').forEach(btn => {
  btn.addEventListener('click', () => {
    tags = btn.dataset.tags.split(',').map(s => s.trim()).filter(Boolean);
    renderTags();
    activatePath('coldstart');
    fetchColdstart();
  });
});


/* ── Path C: Describe It (Conversational) ────────────────────── */
async function fetchChat() {
  const query  = document.getElementById('chat-query').value.trim();
  const userId = document.getElementById('chat-user-id').value.trim() || null;
  if (!query) { document.getElementById('chat-query').focus(); return; }
  renderSkeletons(8);
  try {
    const data = await api.chat(query, userId, 8);
    if (data.llm_summary) renderLLMSummary(data.llm_summary);
    renderCards(data.recommendations ?? [], 'chat', 'your query');
  } catch (err) {
    renderError(err.message);
  }
}

document.getElementById('chat-submit')?.addEventListener('click', fetchChat);
document.getElementById('chat-query')?.addEventListener('keydown', e => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) fetchChat();
});

document.querySelectorAll('.sample-query').forEach(btn => {
  btn.addEventListener('click', () => {
    document.getElementById('chat-query').value = btn.dataset.query;
    activatePath('chat');
    fetchChat();
  });
});


/* ── Intersection observer — animate pipeline/chart on scroll ─── */
const obs = new IntersectionObserver(entries => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      e.target.style.animationPlayState = 'running';
      obs.unobserve(e.target);
    }
  });
}, { threshold: 0.25 });

document.querySelectorAll('.cbar-fill').forEach(el => {
  el.style.animationPlayState = 'paused';
  obs.observe(el);
});


/* ── Init ─────────────────────────────────────────────────────── */
(function init() {
  loadMetrics();
})();
