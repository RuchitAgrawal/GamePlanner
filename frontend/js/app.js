/**
 * app.js — Mode switching, event wiring, live metrics, tag-input.
 *
 * Orchestrates all 3 demo modes (CF, Cold-Start, Conversational)
 * and the live metrics banner in the hero section.
 */

/* ═══════════════════════════════════════════════════════════════
   1. Navbar scroll behaviour
═══════════════════════════════════════════════════════════════ */
window.addEventListener('scroll', () => {
  const nav = document.getElementById('navbar');
  if (nav) nav.classList.toggle('scrolled', window.scrollY > 40);
}, { passive: true });


/* ═══════════════════════════════════════════════════════════════
   2. Live metrics banner (hero section)
═══════════════════════════════════════════════════════════════ */
async function loadMetrics() {
  try {
    const data = await api.metrics();
    const best = data.best_model ?? {};

    const set = (id, val) => {
      const el = document.getElementById(id);
      if (el && val !== undefined) el.textContent = val;
    };

    const modelName = best.model_name ?? best.model ?? 'GMF';
    const hr10 = best.hr_at_10 != null ? (best.hr_at_10 * 100).toFixed(2) + '%' : '30.96%';
    const nd10 = best.ndcg_at_10 != null ? (best.ndcg_at_10 * 100).toFixed(2) + '%' : '18.41%';

    set('mv-model', modelName.toUpperCase());
    set('mv-hr10',  hr10);
    set('mv-ndcg10', nd10);
  } catch (_) {
    // Static fallback values already in HTML — silently ignore
  }
}


/* ═══════════════════════════════════════════════════════════════
   3. Mode tab switching
═══════════════════════════════════════════════════════════════ */
function activateMode(mode) {
  // Update tabs
  document.querySelectorAll('.mode-tab').forEach(tab => {
    const active = tab.dataset.mode === mode;
    tab.classList.toggle('active', active);
    tab.setAttribute('aria-selected', active);
  });

  // Update panels
  document.querySelectorAll('.mode-panel').forEach(panel => {
    panel.classList.toggle('active', panel.id === `panel-${mode}`);
  });

  // Clear previous results
  clearResults();
}

document.querySelectorAll('.mode-tab').forEach(tab => {
  tab.addEventListener('click', () => activateMode(tab.dataset.mode));
});

// Hero CTA buttons scroll + activate mode
document.querySelectorAll('[data-mode]').forEach(btn => {
  if (btn.closest('.hero-ctas')) {
    btn.addEventListener('click', e => {
      const mode = btn.dataset.mode;
      activateMode(mode);
      // small delay to let scroll settle
      setTimeout(() => {
        document.getElementById('demo')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }, 50);
    });
  }
});


/* ═══════════════════════════════════════════════════════════════
   4. Mode A — CF Recommendations
═══════════════════════════════════════════════════════════════ */
async function fetchCF() {
  const userId = document.getElementById('user-id-input').value.trim();
  if (!userId) {
    document.getElementById('user-id-input').focus();
    return;
  }

  renderSkeletons(8);
  document.getElementById('demo').scrollIntoView({ behavior: 'smooth', block: 'start' });

  try {
    const data = await api.recommend(userId);
    const items = data.recommendations ?? [];
    renderCards(items, 'cf', userId);
  } catch (err) {
    renderError(err.message);
  }
}

document.getElementById('cf-submit').addEventListener('click', fetchCF);
document.getElementById('user-id-input').addEventListener('keydown', e => {
  if (e.key === 'Enter') fetchCF();
});

// Sample user ID buttons
document.querySelectorAll('.pill[data-user-id]').forEach(btn => {
  btn.addEventListener('click', () => {
    document.getElementById('user-id-input').value = btn.dataset.userId;
    activateMode('cf');
    fetchCF();
  });
});


/* ═══════════════════════════════════════════════════════════════
   5. Mode B — Cold-Start (tag input)
═══════════════════════════════════════════════════════════════ */
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
    btn.addEventListener('click', () => {
      tags.splice(Number(btn.dataset.idx), 1);
      renderTags();
    });
  });
}

function addTag(value) {
  const trimmed = value.trim().replace(/,$/, '');
  if (trimmed && !tags.includes(trimmed)) {
    tags.push(trimmed);
    renderTags();
  }
}

function escHtmlLocal(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

const tagField = document.getElementById('tag-field');

tagField.addEventListener('keydown', e => {
  if (e.key === 'Enter' || e.key === ',') {
    e.preventDefault();
    addTag(tagField.value);
    tagField.value = '';
  } else if (e.key === 'Backspace' && !tagField.value && tags.length) {
    tags.pop();
    renderTags();
  }
});

// Click anywhere in the box focuses the input
document.getElementById('tag-input-container').addEventListener('click', () => tagField.focus());

async function fetchColdstart() {
  // Also grab anything still typed in the field
  if (tagField.value.trim()) {
    addTag(tagField.value);
    tagField.value = '';
  }
  if (!tags.length) {
    tagField.focus();
    return;
  }

  renderSkeletons(8);
  document.getElementById('demo').scrollIntoView({ behavior: 'smooth', block: 'start' });

  try {
    const data = await api.coldstart(tags);
    const items = data.recommendations ?? [];
    renderCards(items, 'coldstart');
  } catch (err) {
    renderError(err.message);
  }
}

document.getElementById('coldstart-submit').addEventListener('click', fetchColdstart);

// Preset tag buttons
document.querySelectorAll('.sample-tags').forEach(btn => {
  btn.addEventListener('click', () => {
    tags = btn.dataset.tags.split(',').map(s => s.trim()).filter(Boolean);
    renderTags();
    activateMode('coldstart');
    fetchColdstart();
  });
});


/* ═══════════════════════════════════════════════════════════════
   6. Mode C — Conversational
═══════════════════════════════════════════════════════════════ */
async function fetchChat() {
  const query  = document.getElementById('chat-query').value.trim();
  const userId = document.getElementById('chat-user-id').value.trim() || null;
  if (!query) {
    document.getElementById('chat-query').focus();
    return;
  }

  renderSkeletons(8);
  document.getElementById('demo').scrollIntoView({ behavior: 'smooth', block: 'start' });

  try {
    const data  = await api.chat(query, userId);
    const items = data.recommendations ?? [];
    if (data.llm_summary) renderLLMSummary(data.llm_summary, 'chat');
    renderCards(items, 'chat', userId);
  } catch (err) {
    renderError(err.message);
  }
}

document.getElementById('chat-submit').addEventListener('click', fetchChat);
document.getElementById('chat-query').addEventListener('keydown', e => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) fetchChat();
});

// Sample query pills
document.querySelectorAll('.sample-query').forEach(btn => {
  btn.addEventListener('click', () => {
    document.getElementById('chat-query').value = btn.dataset.query;
    activateMode('chat');
    fetchChat();
  });
});


/* ═══════════════════════════════════════════════════════════════
   7. Intersection Observer — animate pipeline steps on scroll
═══════════════════════════════════════════════════════════════ */
const pipeObs = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.style.animationPlayState = 'running';
      pipeObs.unobserve(entry.target);
    }
  });
}, { threshold: 0.3 });

document.querySelectorAll('.pipeline-step, .cbar-fill').forEach(el => {
  el.style.animationPlayState = 'paused';
  pipeObs.observe(el);
});


/* ═══════════════════════════════════════════════════════════════
   8. Init
═══════════════════════════════════════════════════════════════ */
(async function init() {
  loadMetrics(); // non-blocking
})();
