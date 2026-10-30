// Recommender Page Script
document.addEventListener('DOMContentLoaded', () => {
  const searchInput = document.getElementById('rec-search');
  const dropdown = document.getElementById('rec-dropdown');
  const selectedMovieId = document.getElementById('selected-movie-id');
  const seedDisplay = document.getElementById('selected-seed-display');
  const methodSelect = document.getElementById('rec-method');
  const kSelect = document.getElementById('rec-k');
  const fetchBtn = document.getElementById('btn-fetch-recs');

  const loadingBox = document.getElementById('recs-loading');
  const recsGrid = document.getElementById('recs-grid');
  const rulesSection = document.getElementById('rules-section');
  const rulesTbody = document.getElementById('rules-tbody');

  // Parse URL params (e.g. /recommend?movie_id=260)
  const urlParams = new URLSearchParams(window.location.search);
  const paramMovieId = urlParams.get('movie_id');
  if (paramMovieId) {
    selectedMovieId.value = paramMovieId;
  }

  // Initial load
  loadRecommendations();

  // Autocomplete search
  searchInput.addEventListener('input', debounce(async (e) => {
    const q = e.target.value.trim();
    if (!q) {
      dropdown.style.display = 'none';
      return;
    }
    try {
      const res = await fetch(`/api/search?q=${encodeURIComponent(q)}&limit=6`);
      const data = await res.json();
      if (!data.results || data.results.length === 0) {
        dropdown.innerHTML = '<div style="padding: 1rem; color: var(--text-dim); text-align: center;">No matching titles found</div>';
        dropdown.style.display = 'block';
        return;
      }
      dropdown.innerHTML = data.results.map(m => `
        <div class="search-item" onclick="chooseMovie(${m.movie_id}, '${m.title.replace(/'/g, "\\'")}')">
          <div>
            <div class="search-item-title">${m.title}</div>
            <div class="search-item-meta">${m.genres.slice(0, 3).join(', ')}</div>
          </div>
          <div style="text-align: right;">
            <span class="chip" style="color: var(--accent-amber); font-weight: 700;">★ ${m.avg_rating.toFixed(1)}</span>
          </div>
        </div>
      `).join('');
      dropdown.style.display = 'block';
    } catch (err) {
      console.error(err);
    }
  }, 200));

  window.chooseMovie = (id, title) => {
    selectedMovieId.value = id;
    seedDisplay.innerHTML = `<span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: var(--primary);"></span><span>Selected: ${title} [ID: ${id}]</span>`;
    dropdown.style.display = 'none';
    searchInput.value = '';
    loadRecommendations();
  };

  fetchBtn.addEventListener('click', loadRecommendations);

  async function loadRecommendations() {
    const mid = selectedMovieId.value || '1';
    const method = methodSelect.value;
    const k = kSelect.value;

    loadingBox.style.display = 'flex';
    recsGrid.innerHTML = '';
    rulesSection.style.display = 'none';

    try {
      const res = await fetch(`/api/recommend?movie_id=${mid}&method=${method}&k=${k}`);
      if (!res.ok) {
        throw new Error(`API error ${res.status}`);
      }
      const data = await res.json();

      // Update seed display if title present
      if (data.target_movie) {
        seedDisplay.innerHTML = `<span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: var(--primary);"></span><span>Selected: ${data.target_movie.title} [ID: ${data.target_movie.movie_id}]</span>`;
      }

      // Render Recommendations
      if (!data.recommendations || data.recommendations.length === 0) {
        recsGrid.innerHTML = `
          <div class="empty-box" style="grid-column: 1 / -1;">
            <h3 style="color: var(--text-main); margin-bottom: 0.5rem;">No recommendations returned for this algorithm</h3>
            <p>Try switching to another strategy such as Hybrid Ensemble or Content-Based.</p>
          </div>
        `;
      } else {
        recsGrid.innerHTML = data.recommendations.map(r => `
          <div class="rec-card">
            <div>
              <div class="rec-header">
                <div>
                  <div class="rec-title">${r.title} ${r.year ? `<span style="font-weight: 500; font-size: 0.95rem; color: var(--text-muted);">(${r.year})</span>` : ''}</div>
                  <div style="margin-top: 0.45rem; display: flex; flex-wrap: wrap; gap: 0.4rem;">
                    ${r.genres.map(g => `<span class="chip">${g}</span>`).join('')}
                  </div>
                </div>
                <div class="rec-score-badge">
                  <span>${(r.score * 100).toFixed(0)}%</span>
                  <span class="rec-score-label">Match</span>
                </div>
              </div>
              <div class="rec-why">💡 ${r.why}</div>
            </div>

            <div class="rec-breakdown">
              <span style="color: var(--accent-amber); font-weight: 700;">★ ${r.avg_rating.toFixed(1)}</span>
              <span style="color: var(--text-dim);">(${r.n_ratings} ratings)</span>
              <span style="margin: 0 4px; color: var(--border-subtle);">&bull;</span>
              <span>Content: ${(r.component_scores.content * 100).toFixed(0)}%</span>
              <span style="margin: 0 4px; color: var(--border-subtle);">&bull;</span>
              <span>CF: ${(r.component_scores.cf * 100).toFixed(0)}%</span>
              <span style="margin: 0 4px; color: var(--border-subtle);">&bull;</span>
              <span>Rule: ${(r.component_scores.rule * 100).toFixed(0)}%</span>
            </div>
          </div>
        `).join('');
      }

      // Render Association Rules Table if rules used
      if (data.association_rules_used && data.association_rules_used.length > 0) {
        rulesTbody.innerHTML = data.association_rules_used.map(rule => `
          <tr>
            <td style="font-weight: 700; color: var(--text-main);">${rule.antecedent_str}</td>
            <td style="color: var(--primary); font-weight: 600;">${rule.consequent_str}</td>
            <td>${(rule.support * 100).toFixed(1)}%</td>
            <td>${(rule.confidence * 100).toFixed(1)}%</td>
            <td><span class="chip chip-primary" style="font-weight: 800;">${rule.lift.toFixed(2)}x Lift</span></td>
          </tr>
        `).join('');
        rulesSection.style.display = 'block';
      }

    } catch (err) {
      console.error(err);
      recsGrid.innerHTML = `<div class="empty-box" style="grid-column: 1 / -1; color: var(--accent-rose);">Error loading recommendations: ${err.message}</div>`;
    } finally {
      loadingBox.style.display = 'none';
    }
  }

  // Close dropdown on outside click
  document.addEventListener('click', (e) => {
    if (!searchInput.contains(e.target) && !dropdown.contains(e.target)) {
      dropdown.style.display = 'none';
    }
  });
});
