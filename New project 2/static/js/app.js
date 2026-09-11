(function () {
  /* ── Chart.js dark theme defaults ── */
  function applyChartDefaults() {
    if (typeof Chart === "undefined") return;
    Chart.defaults.color = "#94a3b8";
    Chart.defaults.borderColor = "rgba(255,255,255,0.06)";
    Chart.defaults.plugins.legend.labels.padding = 16;
    Chart.defaults.plugins.legend.labels.usePointStyle = true;
    Chart.defaults.plugins.legend.labels.pointStyleWidth = 10;
    Chart.defaults.scale.grid = Chart.defaults.scale.grid || {};
    Chart.defaults.scale.grid.color = "rgba(255,255,255,0.05)";
  }

  function jsonBlock(value) {
    return JSON.stringify(value, null, 2);
  }

  function setFeedSource(image, baseSrc, cameraIndex) {
    if (!image || !baseSrc) {
      return;
    }
    const separator = baseSrc.includes("?") ? "&" : "?";
    image.src = `${baseSrc}${separator}camera=${encodeURIComponent(cameraIndex)}&t=${Date.now()}`;
  }

  function initPreviewFeed() {
    const preview = document.getElementById("previewFeed");
    const cameraInput = document.getElementById("cameraIndex");
    if (!preview || !cameraInput) {
      return;
    }
    setFeedSource(preview, preview.dataset.baseSrc, cameraInput.value || 0);
    cameraInput.addEventListener("change", function () {
      setFeedSource(preview, preview.dataset.baseSrc, cameraInput.value || 0);
    });
  }

  async function postJson(url, payload) {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload || {}),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Request failed");
    }
    return data;
  }

  function initRegistrationActions() {
    const captureButton = document.getElementById("captureButton");
    const trainButton = document.getElementById("trainButton");
    const cameraInput = document.getElementById("cameraIndex");
    const countInput = document.getElementById("captureCount");
    const output = document.getElementById("captureOutput");

    if (captureButton && output) {
      captureButton.addEventListener("click", async function () {
        captureButton.disabled = true;
        output.textContent = "⏳ Capturing face images...";
        try {
          const data = await postJson(captureButton.dataset.url, {
            camera: Number(cameraInput.value || 0),
            count: Number(countInput.value || 30),
          });
          output.textContent = "✅ Capture complete!\n\n" + jsonBlock(data);
        } catch (error) {
          output.textContent = "❌ Error: " + error.message;
        } finally {
          captureButton.disabled = false;
        }
      });
    }

    if (trainButton && output) {
      trainButton.addEventListener("click", async function () {
        trainButton.disabled = true;
        output.textContent = "⏳ Training encodings...";
        try {
          const data = await postJson(trainButton.dataset.url, {});
          output.textContent = "✅ Training complete!\n\n" + jsonBlock(data);
        } catch (error) {
          output.textContent = "❌ Error: " + error.message;
        } finally {
          trainButton.disabled = false;
        }
      });
    }
  }

  function initAttendanceFeed() {
    const startButton = document.getElementById("startAttendanceFeed");
    const image = document.getElementById("attendanceFeed");
    const cameraInput = document.getElementById("attendanceCameraIndex");
    if (!startButton || !image || !cameraInput) {
      return;
    }
    startButton.addEventListener("click", function () {
      setFeedSource(image, image.dataset.baseSrc, cameraInput.value || 0);
    });
  }

  function chartAvailable() {
    return typeof window.Chart !== "undefined";
  }

  function emptySummary(summary) {
    return !Array.isArray(summary) || summary.length === 0;
  }

  /* gradient bar helper */
  function accentGradient(ctx) {
    var gradient = ctx.createLinearGradient(0, 0, 0, 300);
    gradient.addColorStop(0, "rgba(99, 102, 241, 0.85)");
    gradient.addColorStop(1, "rgba(139, 92, 246, 0.25)");
    return gradient;
  }

  function initDashboardChart() {
    var canvas = document.getElementById("dashboardBehaviorChart");
    if (!canvas || !chartAvailable()) {
      return;
    }
    var summary = window.dashboardBehaviorSummary || [];
    var labels = emptySummary(summary) ? ["No data"] : summary.map(function (row) { return row.roll_no; });
    var scores = emptySummary(summary) ? [0] : summary.map(function (row) { return row.average_attention; });

    new Chart(canvas, {
      type: "bar",
      data: {
        labels: labels,
        datasets: [
          {
            label: "Average Attention",
            data: scores,
            backgroundColor: accentGradient(canvas.getContext("2d")),
            borderRadius: 8,
            borderSkipped: false,
          },
        ],
      },
      options: {
        maintainAspectRatio: false,
        plugins: { legend: { display: true, position: "top" } },
        scales: {
          y: { beginAtZero: true, max: 100 },
          x: { grid: { display: false } },
        },
      },
    });
  }

  function initBehaviorCharts() {
    var scoreCanvas = document.getElementById("behaviorScoreChart");
    var statusCanvas = document.getElementById("behaviorStatusChart");
    if ((!scoreCanvas && !statusCanvas) || !chartAvailable()) {
      return;
    }

    var summary = window.behaviorSummary || [];
    var labels = emptySummary(summary) ? ["No data"] : summary.map(function (row) { return "" + row.roll_no; });
    var scores = emptySummary(summary) ? [0] : summary.map(function (row) { return row.average_attention; });

    if (scoreCanvas) {
      new Chart(scoreCanvas, {
        type: "bar",
        data: {
          labels: labels,
          datasets: [
            {
              label: "Average Attention",
              data: scores,
              backgroundColor: function (ctx) {
                var g = scoreCanvas.getContext("2d").createLinearGradient(0, 0, 0, 300);
                g.addColorStop(0, "rgba(34, 197, 94, 0.85)");
                g.addColorStop(1, "rgba(34, 197, 94, 0.15)");
                return g;
              }(),
              borderRadius: 8,
              borderSkipped: false,
            },
          ],
        },
        options: {
          maintainAspectRatio: false,
          plugins: { legend: { display: true, position: "top" } },
          scales: {
            y: { beginAtZero: true, max: 100 },
            x: { grid: { display: false } },
          },
        },
      });
    }

    if (statusCanvas) {
      var totals = summary.reduce(
        function (accumulator, row) {
          accumulator.attentive += row.attentive || 0;
          accumulator.distracted += row.distracted || 0;
          accumulator.sleepy += row.sleepy || 0;
          accumulator.unknown += row.unknown || 0;
          return accumulator;
        },
        { attentive: 0, distracted: 0, sleepy: 0, unknown: 0 }
      );

      var data = emptySummary(summary)
        ? [0, 0, 0, 0]
        : [totals.attentive, totals.distracted, totals.sleepy, totals.unknown];

      new Chart(statusCanvas, {
        type: "doughnut",
        data: {
          labels: ["Attentive", "Distracted", "Sleepy", "Unknown"],
          datasets: [
            {
              data: data,
              backgroundColor: [
                "rgba(34, 197, 94, 0.8)",
                "rgba(245, 158, 11, 0.8)",
                "rgba(239, 68, 68, 0.8)",
                "rgba(100, 116, 139, 0.5)"
              ],
              borderColor: "transparent",
              borderWidth: 0,
              hoverOffset: 6,
            },
          ],
        },
        options: {
          maintainAspectRatio: false,
          cutout: "65%",
          plugins: {
            legend: { position: "bottom", labels: { padding: 20 } },
          },
        },
      });
    }
  }

  document.addEventListener("DOMContentLoaded", function () {
    applyChartDefaults();
    initPreviewFeed();
    initRegistrationActions();
    initAttendanceFeed();
    initDashboardChart();
    initBehaviorCharts();
  });
})();
