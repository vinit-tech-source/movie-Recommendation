// Predict Page Script
document.addEventListener('DOMContentLoaded', () => {
  // Genre pills selection
  const pills = document.querySelectorAll('.genre-pill');
  pills.forEach(p => {
    p.addEventListener('click', () => {
      p.classList.toggle('selected');
    });
  });

  // Rating Predictor Form
  const ratingForm = document.getElementById('rating-form');
  const ratingOutput = document.getElementById('rating-output');
  const classBadge = document.getElementById('output-class-badge');
  const ratingValue = document.getElementById('output-rating-value');
  const probContainer = document.getElementById('output-probabilities');

  ratingForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const selectedGenres = Array.from(document.querySelectorAll('.genre-pill.selected'))
      .map(el => el.getAttribute('data-genre'));

    const year = parseInt(document.getElementById('pred-year').value, 10) || 2022;
    const nRatings = parseInt(document.getElementById('pred-ratings').value, 10) || 50;
    const nTags = parseInt(document.getElementById('pred-tags').value, 10) || 0;

    const payload = {
      genres: selectedGenres,
      year: year,
      n_ratings: nRatings,
      n_tags: nTags
    };

    try {
      const res = await fetch('/api/predict-rating', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Prediction failed');

      // Update badge
      const cls = data.predicted_class;
      classBadge.textContent = cls;
      classBadge.className = '';
      if (cls === 'Low') classBadge.classList.add('badge-low');
      else if (cls === 'Medium') classBadge.classList.add('badge-medium');
      else classBadge.classList.add('badge-high');

      // Update continuous score
      ratingValue.textContent = `★ ${data.predicted_rating_value.toFixed(2)} / 5.0`;

      // Probabilities progress bars
      const probs = data.class_probabilities || {};
      probContainer.innerHTML = Object.entries(probs).map(([k, p]) => `
        <div style="margin-bottom: 0.5rem;">
          <div style="display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 2px;">
            <span>${k}</span>
            <span>${(p * 100).toFixed(1)}%</span>
          </div>
          <div class="progress-bar-container">
            <div class="progress-bar-fill" style="width: ${(p * 100).toFixed(1)}%;"></div>
          </div>
        </div>
      `).join('');

      ratingOutput.style.display = 'block';
    } catch (err) {
      alert(`Prediction Error: ${err.message}`);
    }
  });

  // Genre Classifier Form
  const genreForm = document.getElementById('genre-form');
  const genreText = document.getElementById('genre-text');
  const genreOutput = document.getElementById('genre-output');
  const genreList = document.getElementById('genre-predictions-list');

  genreForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const text = genreText.value.trim();
    if (!text) {
      alert('Please enter a plot synopsis');
      return;
    }

    try {
      const res = await fetch('/api/predict-genre', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Classification failed');

      genreList.innerHTML = (data.top_predictions || []).map((pred, idx) => `
        <div style="margin-bottom: 0.75rem;">
          <div style="display: flex; justify-content: space-between; font-size: 0.9rem; font-weight: 600; margin-bottom: 3px;">
            <span>#${idx + 1} ${pred.genre}</span>
            <span style="color: var(--accent-blue);">${(pred.probability * 100).toFixed(1)}%</span>
          </div>
          <div class="progress-bar-container">
            <div class="progress-bar-fill" style="width: ${(pred.probability * 100).toFixed(1)}%;"></div>
          </div>
        </div>
      `).join('');

      genreOutput.style.display = 'block';
    } catch (err) {
      alert(`Classification Error: ${err.message}`);
    }
  });

  // Quick Sample texts helper
  window.setSample = (type) => {
    const samples = {
      scifi: 'In the distant future, an astronaut travels through a mysterious interstellar wormhole in a desperate attempt to find a new habitable planet for dying humanity.',
      comedy: 'Two bumbling wedding videographers accidentally record a mob execution and must impersonate eccentric French chefs to escape the pursuing hitmen.',
      doc: 'An investigative journey across remote polar regions capturing the ecological struggle of wild penguin colonies against changing climate patterns.'
    };
    if (samples[type]) {
      genreText.value = samples[type];
      genreForm.dispatchEvent(new Event('submit'));
    }
  };
});

