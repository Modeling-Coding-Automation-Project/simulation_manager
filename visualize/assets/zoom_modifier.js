/**
 * zoom_modifier.js
 *
 * Custom zoom interactions for Plotly/Dash waveform plots.
 *
 *   drag             → XY zoom  (Plotly 'zoom' dragmode)
 *   Shift + drag     → pan     (Plotly 'pan' dragmode)
 *   wheel            → X-only zoom
 *   Alt   + wheel    → Y-only zoom
 *   double-click     → reset zoom (Plotly built-in)
 */
(function() {
'use strict';

/* ------------------------------------------------------------------
 * Keyboard-driven dragmode switching
 * ------------------------------------------------------------------ */
var currentModifier = null;

function getPlotlyGraphs() {
  return Array.from(document.querySelectorAll('.js-plotly-plot'));
}

function setAllDragMode(mode) {
  getPlotlyGraphs().forEach(function(gd) {
    if (gd._fullLayout && gd._fullLayout.dragmode !== mode) {
      Plotly.relayout(gd, {dragmode: mode});
    }
  });
}

document.addEventListener('keyup', function(e) {
  if ((currentModifier === 'shift' && !e.shiftKey) ||
      (currentModifier === 'ctrl' && !e.ctrlKey)) {
    currentModifier = null;
    setAllDragMode('zoom');
  }
});

/* Restore dragmode when window loses focus (e.g. Alt-Tab) */
window.addEventListener('blur', function() {
  if (currentModifier !== null) {
    currentModifier = null;
    setAllDragMode('zoom');
  }
});

/* ------------------------------------------------------------------
 * Per-graph wheel zoom
 * ------------------------------------------------------------------ */

/**
 * Returns the mouse Y position normalised to Plotly domain coordinates
 * (0 = bottom of plot area, 1 = top) for the given graph div and event.
 */
function getNormMouseY(gd, e) {
  var layout = gd._fullLayout;
  var margin = layout.margin || {l: 0, r: 0, t: 0, b: 0};
  var plotHeight = layout.height - margin.t - margin.b;
  var rect = gd.getBoundingClientRect();
  var mouseY = e.clientY - rect.top - margin.t;
  /* Plotly domain: 0 = bottom of plot area, 1 = top */
  return 1 - mouseY / plotHeight;
}

/**
 * Builds Plotly relayout updates to zoom axes matching axisPattern.
 * When normY is provided (Alt+wheel), only the yaxis whose domain
 * contains that normalised coordinate is updated.
 */
function buildRangeUpdates(layout, axisPattern, factor, normY) {
  var updates = {};
  var found = false;
  Object.keys(layout).forEach(function(key) {
    if (axisPattern.test(key) && layout[key] && layout[key].range) {
      /* For Y-axis zoom: skip axes whose domain does not contain the cursor */
      if (normY !== undefined) {
        var domain = layout[key].domain || [0, 1];
        if (normY < domain[0] || normY > domain[1]) {
          return;
        }
      }
      var r = layout[key].range;
      var mid = (r[0] + r[1]) / 2;
      var half = (r[1] - r[0]) / 2 * factor;
      updates[key + '.range'] = [mid - half, mid + half];
      found = true;
    }
  });
  return found ? updates : null;
}

function setupWheelZoom(gd) {
  if (gd._customWheelAttached) return;
  gd._customWheelAttached = true;

  gd.addEventListener('wheel', function(e) {
    if (!gd._fullLayout) return;
    e.preventDefault();
    e.stopImmediatePropagation();

    /* Scroll up (deltaY < 0) → zoom in (factor < 1) */
    var factor = e.deltaY > 0 ? 1.2 : (1 / 1.2);
    var pattern = e.altKey ? /^yaxis\d*$/ : /^xaxis\d*$/;
    /* Alt+wheel: pass normalised mouse Y so only the hovered subplot is zoomed
     */
    var normY = e.altKey ? getNormMouseY(gd, e) : undefined;
    var updates = buildRangeUpdates(gd._fullLayout, pattern, factor, normY);
    if (updates) {
      Plotly.relayout(gd, updates);
    }
  }, {capture: true, passive: false});
}

/* ------------------------------------------------------------------
 * MutationObserver – attach listeners when graphs appear in DOM
 * ------------------------------------------------------------------ */
function scanAndSetup() {
  document.querySelectorAll('.js-plotly-plot').forEach(setupWheelZoom);
}

var observer = new MutationObserver(scanAndSetup);
observer.observe(document.documentElement, {childList: true, subtree: true});

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', scanAndSetup);
} else {
  scanAndSetup();
}
setTimeout(scanAndSetup, 500);
setTimeout(scanAndSetup, 1500);
}());
