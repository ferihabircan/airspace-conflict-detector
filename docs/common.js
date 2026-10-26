// Shared helpers for the portfolio project pages: theme toggle and Plotly styling.
(function () {
  const root = document.documentElement;
  try {
    const saved = localStorage.getItem("theme");
    if (saved) root.dataset.theme = saved;
  } catch (e) { /* storage unavailable */ }

  window.isDark = () =>
    root.dataset.theme === "dark" ||
    (!root.dataset.theme && matchMedia("(prefers-color-scheme: dark)").matches);

  const redrawers = [];
  window.onThemeChange = (fn) => { redrawers.push(fn); };

  document.addEventListener("DOMContentLoaded", () => {
    const btn = document.querySelector(".theme-toggle");
    if (!btn) return;
    const label = () => { btn.textContent = isDark() ? "☀︎" : "☾"; };
    label();
    btn.addEventListener("click", () => {
      root.dataset.theme = isDark() ? "light" : "dark";
      try { localStorage.setItem("theme", root.dataset.theme); } catch (e) {}
      label();
      redrawers.forEach((fn) => fn());
    });
  });

  window.css = (name) => getComputedStyle(root).getPropertyValue(name).trim();

  // Base Plotly layout that follows the page theme.
  window.baseLayout = (extra = {}) => {
    const text = css("--text"), muted = css("--muted"), grid = css("--border");
    const axis = { gridcolor: grid, zerolinecolor: grid, linecolor: grid,
                   tickfont: { color: muted, size: 12 }, titlefont: { color: muted, size: 12.5 } };
    const { xaxis = {}, yaxis = {}, ...rest } = extra;
    return Object.assign({
      paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
      font: { family: "Inter, system-ui, sans-serif", color: text, size: 13 },
      margin: { l: 56, r: 16, t: 36, b: 48 },
      xaxis: Object.assign({}, axis, xaxis), yaxis: Object.assign({}, axis, yaxis),
      legend: { orientation: "h", y: 1.12, x: 0, font: { color: muted } },
      hoverlabel: { font: { family: "Inter, system-ui, sans-serif" } },
    }, rest);
  };
  window.plotConfig = { displayModeBar: false, responsive: true };
})();
