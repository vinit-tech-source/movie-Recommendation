// Trends & Time Series Forecasting Page Script
document.addEventListener('DOMContentLoaded', () => {
  const sourceSelect = document.getElementById('trend-source');
  const typeSelect = document.getElementById('trend-type');
  const typeGroup = document.getElementById('type-filter-group');
  const genreSelect = document.getElementById('trend-genre');

  const chartTitle = document.getElementById('chart-title');
  const modelBadge = document.getElementById('model-order-badge');
  const loadingBox = document.getElementById('trend-loading');
  const chartWrapper = document.getElementById('chart-wrapper');

  const diagAdf = document.getElementById('diag-adf');
  const diagRmse = document.getElementById('diag-rmse');
  const diagNaive = document.getElementById('diag-naive');
  const diagMsg = document.getElementById('diag-msg');

  let mainChart = null;
  let genresChart = null;
  let histChart = null;

  const NETFLIX_GENRES = [
    'Action & Adventure', 'Comedies', 'Dramas', 'Documentaries',
    'International TV Shows', 'Children & Family Movies', 'Crime TV Shows',
    'Horror Movies', 'Romantic Movies', 'Sci-Fi & Fantasy', 'Thrillers'
  ];
  const MOVIELENS_GENRES = [
    'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
    'Documentary', 'Drama', 'Fantasy', 'Horror', 'Mystery', 'Romance',
    'Sci-Fi', 'Thriller', 'War', 'Western'
  ];

  function populateGenreDropdown(source) {
    const list = source === 'netflix' ? NETFLIX_GENRES : MOVIELENS_GENRES;
    genreSelect.innerHTML = '<option value="" selected>All Categories</option>' +
      list.map(g => `<option value="${g}">${g}</option>`).join('');
  }

  // Initial population
  populateGenreDropdown('netflix');
  loadMainTrend();
  loadWarehouseStats();

  sourceSelect.addEventListener('change', () => {
    const src = sourceSelect.value;
    if (src === 'netflix') {
      typeGroup.style.display = 'block';
    } else {
      typeGroup.style.display = 'none';
      typeSelect.value = '';
    }
    populateGenreDropdown(src);
    loadMainTrend();
  });

  typeSelect.addEventListener('change', loadMainTrend);
  genreSelect.addEventListener('change', loadMainTrend);

  async function loadMainTrend() {
    const src = sourceSelect.value;
    const type = typeSelect.value;
    const genre = genreSelect.value;

    chartWrapper.style.opacity = '0.3';
    loadingBox.style.display = 'flex';

    let url = `/api/trends?source=${src}`;
    if (type) url += `&type=${encodeURIComponent(type)}`;
    if (genre) url += `&genre=${encodeURIComponent(genre)}`;

    try {
      const res = await fetch(url);
      const data = await res.json();

      // Update Diagnostics
      if (data.model_order) {
        modelBadge.textContent = `ARIMA(${data.model_order.join(', ')})`;
      } else {
        modelBadge.textContent = 'History Only';
      }

      chartTitle.textContent = `${src === 'netflix' ? 'Netflix Monthly Additions' : 'MovieLens Monthly Ratings'}${genre ? ` - ${genre}` : ''}${type ? ` (${type})` : ''}`;

      if (data.adf_pvalue !== undefined) {
        diagAdf.textContent = `Stationary (d=1, p = ${data.adf_pvalue < 0.0001 ? '< 0.0001' : data.adf_pvalue.toFixed(4)})`;
      } else {
        diagAdf.textContent = 'N/A';
      }

      if (data.test_rmse) {
        diagRmse.textContent = `${data.test_rmse}`;
        diagNaive.textContent = `(vs Naive ${data.naive_rmse || '--'})`;
      } else {
        diagRmse.textContent = 'N/A';
        diagNaive.textContent = '';
      }

      diagMsg.textContent = data.message || 'Complete';

      renderMainChart(data);

    } catch (err) {
      console.error(err);
    } finally {
      loadingBox.style.display = 'none';
      chartWrapper.style.opacity = '1';
    }
  }

  function renderMainChart(data) {
    const ctx = document.getElementById('trendChart').getContext('2d');
    if (mainChart) mainChart.destroy();

    const history = data.history || [];
    const forecast = data.forecast || [];

    const histLabels = history.map(h => h.date);
    const fcLabels = forecast.map(f => f.date);
    const allLabels = [...histLabels, ...fcLabels];

    const histData = history.map(h => h.value);
    // Pad forecast with nulls for history length
    const fcData = new Array(histData.length).fill(null);
    if (forecast.length > 0 && histData.length > 0) {
      // Connect last history point to first forecast
      fcData[fcData.length - 1] = histData[histData.length - 1];
    }
    forecast.forEach(f => fcData.push(f.forecast));

    const upperCI = new Array(histData.length).fill(null);
    const lowerCI = new Array(histData.length).fill(null);
    if (forecast.length > 0 && histData.length > 0) {
      upperCI[upperCI.length - 1] = histData[histData.length - 1];
      lowerCI[lowerCI.length - 1] = histData[histData.length - 1];
    }
    forecast.forEach(f => {
      upperCI.push(f.upper_ci);
      lowerCI.push(f.lower_ci);
    });

    const datasets = [
      {
        label: 'Historical Data',
        data: [...histData, ...new Array(forecast.length).fill(null)],
        borderColor: '#2563eb',
        backgroundColor: 'rgba(37, 99, 235, 0.08)',
        borderWidth: 2.2,
        tension: 0.2,
        pointRadius: 1,
      }
    ];

    if (forecast.length > 0) {
      datasets.push({
        label: '12-Month ARIMA Forecast',
        data: fcData,
        borderColor: '#e11d48',
        borderDash: [6, 4],
        borderWidth: 2.5,
        tension: 0.2,
        pointRadius: 2,
        fill: false,
      });

      datasets.push({
        label: 'Upper 95% CI',
        data: upperCI,
        borderColor: 'transparent',
        backgroundColor: 'rgba(225, 29, 72, 0.08)',
        pointRadius: 0,
        fill: '+1',
      });

      datasets.push({
        label: 'Lower 95% CI',
        data: lowerCI,
        borderColor: 'transparent',
        backgroundColor: 'transparent',
        pointRadius: 0,
        fill: false,
      });
    }

    mainChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: allLabels,
        datasets: datasets,
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: {
            labels: { color: '#334155', boxWidth: 14, font: { weight: 600 } }
          },
          tooltip: {
            backgroundColor: '#ffffff',
            titleColor: '#0f172a',
            bodyColor: '#334155',
            borderColor: '#e2e8f0',
            borderWidth: 1,
            boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
            padding: 10,
          }
        },
        scales: {
          x: {
            grid: { color: '#f1f5f9' },
            ticks: { color: '#64748b', maxTicksLimit: 14 }
          },
          y: {
            grid: { color: '#f1f5f9' },
            ticks: { color: '#64748b' }
          }
        }
      }
    });
  }

  async function loadWarehouseStats() {
    try {
      const res = await fetch('/api/stats');
      const data = await res.json();

      // 1. Top Genres Bar Chart
      const tg = data.top_genres || [];
      const tgCtx = document.getElementById('topGenresChart').getContext('2d');
      genresChart = new Chart(tgCtx, {
        type: 'bar',
        data: {
          labels: tg.map(t => t.genre),
          datasets: [{
            label: 'Total Ratings',
            data: tg.map(t => t.count),
            backgroundColor: 'rgba(37, 99, 235, 0.85)',
            borderColor: '#2563eb',
            borderWidth: 1,
            borderRadius: 6,
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            x: {
              grid: { display: false },
              ticks: { color: '#64748b', maxRotation: 45, minRotation: 45, font: { size: 10, weight: 600 } }
            },
            y: {
              grid: { color: '#f1f5f9' },
              ticks: { color: '#64748b' }
            }
          }
        }
      });

      // 2. Rating Histogram Chart
      const rh = data.rating_histogram || [];
      const rhCtx = document.getElementById('ratingHistChart').getContext('2d');
      histChart = new Chart(rhCtx, {
        type: 'bar',
        data: {
          labels: rh.map(r => `★ ${r.rating}`),
          datasets: [{
            label: 'Rating Count',
            data: rh.map(r => r.count),
            backgroundColor: 'rgba(5, 150, 105, 0.85)',
            borderColor: '#059669',
            borderWidth: 1,
            borderRadius: 6,
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            x: {
              grid: { display: false },
              ticks: { color: '#64748b', font: { size: 11, weight: 600 } }
            },
            y: {
              grid: { color: '#f1f5f9' },
              ticks: { color: '#64748b' }
            }
          }
        }
      });

    } catch (err) {
      console.error(err);
    }
  }
});

