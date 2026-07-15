/**
 * api.js — All calls to the GamePlanner FastAPI backend.
 *
 * API_BASE auto-detects from window.location.origin so this works
 * both in local dev (localhost:8000) and on Render (gameplanner.onrender.com).
 */

const API_BASE = window.location.origin;

const api = {

  /** GET /api/v1/health */
  health: async () => {
    const r = await fetch(`${API_BASE}/api/v1/health`);
    if (!r.ok) throw new Error(`Health check failed: ${r.status}`);
    return r.json();
  },

  /** GET /api/v1/metrics */
  metrics: async () => {
    const r = await fetch(`${API_BASE}/api/v1/metrics`);
    if (!r.ok) throw new Error(`Metrics failed: ${r.status}`);
    return r.json();
  },

  /**
   * GET /api/v1/recommend/{user_id}/explain?k=N
   * Returns recommendations with LLM explanations.
   */
  recommend: async (userId, k = 8) => {
    const r = await fetch(`${API_BASE}/api/v1/recommend/${encodeURIComponent(userId)}/explain?k=${k}`);
    if (r.status === 404) throw new Error(`User "${userId}" not found in training data. Try a different ID.`);
    if (!r.ok) {
      const body = await r.json().catch(() => ({}));
      throw new Error(body.detail || `Request failed (${r.status})`);
    }
    return r.json();
  },

  /**
   * POST /api/v1/coldstart
   * Body: { liked_games: string[], k: number }
   */
  coldstart: async (likedGames, k = 8) => {
    const r = await fetch(`${API_BASE}/api/v1/coldstart`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ liked_games: likedGames, k }),
    });
    if (!r.ok) {
      const body = await r.json().catch(() => ({}));
      throw new Error(body.detail || `Request failed (${r.status})`);
    }
    return r.json();
  },

  /**
   * POST /api/v1/chat
   * Body: { query: string, user_id?: string|null, k: number }
   */
  chat: async (query, userId = null, k = 8) => {
    const r = await fetch(`${API_BASE}/api/v1/chat`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ query, user_id: userId || null, k }),
    });
    if (!r.ok) {
      const body = await r.json().catch(() => ({}));
      throw new Error(body.detail || `Request failed (${r.status})`);
    }
    return r.json();
  },
};
