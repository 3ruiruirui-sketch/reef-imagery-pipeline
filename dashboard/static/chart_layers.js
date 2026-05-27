/**
 * chart_layers.js — IHO S-4 Nautical Chart Rendering for Leaflet
 * ================================================================
 * Provides IHO depth colour scheme, isobath polylines, depth soundings,
 * danger zones and nautical symbology for the Carta Náutica dashboard panel.
 *
 * IHO S-4 depth colour scheme (standard for nautical charts):
 *   0–5m   : Very dark navy  (#000080 → #000050)
 *   5–10m  : Dark blue       (#0000AA)
 *   10–20m : Blue            (#0050BF)
 *   20–30m : Medium blue     (#1E7FE0)
 *   30–50m : Light blue      (#7FAEF5)
 *   >50m   : Very light      (#C8E0FF)
 *   Land   : Ochre/brown     (#B5A679)
 */

(function () {
  "use strict";

  // ── IHO S-4 Colour Bands ──────────────────────────────────────────────────────

  /**
   * IHO depth colour bands — 5 ranges + land
   * Each entry: [minDepth, maxDepth, hexColor]
   */
  const IHO_BANDS = [
    { min: 0,   max: 5,   color: "#000870", label: "0–5 m" },
    { min: 5,   max: 10,  color: "#00339A", label: "5–10 m" },
    { min: 10,  max: 20,  color: "#005FBF", label: "10–20 m" },
    { min: 20,  max: 30,  color: "#1E90FF", label: "20–30 m" },
    { min: 30,  max: 50,  color: "#7FBEFF", label: "30–50 m" },
    { min: 50,  max: Infinity, color: "#C8E5FF", label: "> 50 m" },
  ];

  const IHO_LAND_COLOR = "#C8B87A";

  /**
   * depthToColor(depth_m) → CSS hex colour string
   * Returns the IHO S-4 band colour for a given depth in metres.
   */
  function depthToColor(depth_m) {
    if (depth_m < 0 || depth_m === null || depth_m === undefined) return IHO_LAND_COLOR;
    for (const band of IHO_BANDS) {
      if (depth_m < band.max) return band.color;
    }
    return IHO_BANDS[IHO_BANDS.length - 1].color;
  }

  /**
   * depthToRgba(depth_m, alpha) → CSS rgba() string
   */
  function depthToRgba(depth_m, alpha) {
    const hex = depthToColor(depth_m);
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r},${g},${b},${alpha})`;
  }

  /**
   * renderDepthBandsOnCanvas(canvas, depthArray2D, bounds)
   * Paints IHO S-4 colour bands onto a canvas element from a 2D depth array.
   *
   * @param {HTMLCanvasElement} canvas
   * @param {number[][]} depthArray — 2D array [row][col] of depth values in metres
   * @param {object} bounds — { minLon, minLat, maxLon, maxLat }
   * @param {object} options — { landAtZero: bool }
   */
  function renderDepthBandsOnCanvas(canvas, depthArray, bounds, options = {}) {
    const { landAtZero = true } = options;
    const H = depthArray.length;
    const W = depthArray[0] ? depthArray[0].length : 0;
    canvas.width = W;
    canvas.height = H;
    const ctx = canvas.getContext("2d");

    for (let r = 0; r < H; r++) {
      for (let c = 0; c < W; c++) {
        const depth = depthArray[r][c];
        const isLand = landAtZero && depth === 0;
        ctx.fillStyle = isLand ? IHO_LAND_COLOR : depthToColor(depth);
        ctx.fillRect(c, r, 1, 1);
      }
    }
  }

  /**
   * Build IHO depth colour legend HTML element.
   * Appends to map._controlContainer or a given parent element.
   *
   * @param {HTMLElement} parent
   * @returns {HTMLElement} legend container
   */
  function buildDepthLegend(parent) {
    const container = document.createElement("div");
    container.className = "iho-legend";
    container.style.cssText = `
      background: rgba(12, 20, 40, 0.92);
      border: 1px solid rgba(255,255,255,0.15);
      border-radius: 8px;
      padding: 10px 14px;
      font-family: 'Inter', sans-serif;
      font-size: 11px;
      color: #e0ecff;
      line-height: 1.6;
      min-width: 140px;
    `;

    const title = document.createElement("div");
    title.style.cssText = "font-weight:700;letter-spacing:0.5px;margin-bottom:8px;font-size:12px;color:#7ec8e3;";
    title.textContent = "Profundidade (IHO S-4)";
    container.appendChild(title);

    IHO_BANDS.forEach(band => {
      const row = document.createElement("div");
      row.style.cssText = "display:flex;align-items:center;gap:8px;margin-bottom:4px;";

      const swatch = document.createElement("div");
      swatch.style.cssText = `width:22px;height:14px;background:${band.color};border-radius:2px;border:1px solid rgba(255,255,255,0.2);flex-shrink:0;`;

      const label = document.createElement("span");
      label.textContent = band.label;

      row.appendChild(swatch);
      row.appendChild(label);
      container.appendChild(row);
    });

    // Land swatch
    const landRow = document.createElement("div");
    landRow.style.cssText = "display:flex;align-items:center;gap:8px;margin-top:6px;";
    const landSwatch = document.createElement("div");
    landSwatch.style.cssText = `width:22px;height:14px;background:${IHO_LAND_COLOR};border-radius:2px;border:1px solid rgba(255,255,255,0.2);flex-shrink:0;`;
    const landLabel = document.createElement("span");
    landLabel.textContent = "Terra";
    landRow.appendChild(landSwatch);
    landRow.appendChild(landLabel);
    container.appendChild(landRow);

    if (parent) parent.appendChild(container);
    return container;
  }

  // ── Isobath Layer Management ─────────────────────────────────────────────────

  /**
   * Fetch IH/DGRM isobath polylines via the dashboard API proxy.
   * Returns a Leaflet L.geoJSON layer styled per IHO depth conventions.
   *
   * @param {L.Map} map
   * @param {object} bounds — { minLon, minLat, maxLon, maxLat }
   * @param {number[]} depths — e.g. [10, 20, 30] — which isobaths to fetch
   * @returns {Promise<L.GeoJSON>}
   */
  async function fetchIsobathLayer(map, bounds, depths = [10, 20, 30]) {
    const params = new URLSearchParams({
      minlon: bounds.minLon,
      minlat: bounds.minLat,
      maxlon: bounds.maxLon,
      maxlat: bounds.maxLat,
      depths: depths.join(","),
    });

    const url = `/api/isobaths?${params.toString()}`;

    // Isobath IHO style table by depth
    const isobathStyles = {
      0:   { color: "#001050", weight: 3, dashArray: null },
      2:   { color: "#001870", weight: 3, dashArray: null },
      10:  { color: "#003FBF", weight: 3, dashArray: null },
      20:  { color: "#1E7FE0", weight: 3, dashArray: null },
      30:  { color: "#5EAEF5", weight: 3, dashArray: null },
      50:  { color: "#9DC8FF", weight: 2, dashArray: "6,4" },
      100: { color: "#C8E5FF", weight: 2, dashArray: "6,4" },
    };

    const getStyle = (feature) => {
      const depth = parseInt(feature.properties?.Depth || feature.properties?.depth || 0);
      const s = isobathStyles[depth] || { color: "#ffffff", weight: 2, dashArray: null };
      return {
        color: s.color,
        weight: s.weight,
        dashArray: s.dashArray,
        opacity: 0.85,
        fillOpacity: 0,
      };
    };

    return new Promise((resolve, reject) => {
      fetch(url)
        .then(res => {
          if (!res.ok) throw new Error(`isobaths API ${res.status}`);
          return res.json();
        })
        .then(geojson => {
          const layer = L.geoJSON(geojson, {
            style: getStyle,
            onEachFeature: (feature, layer) => {
              const d = feature.properties?.Depth || feature.properties?.depth || "?";
              layer.bindTooltip(`${d}m`, {
                permanent: false,
                direction: "center",
                className: "iho-tooltip",
              });
            },
          });
          resolve(layer);
        })
        .catch(reject);
    });
  }

  // ── Depth Sounding Markers ─────────────────────────────────────────────────

  /**
   * Fetch n depth soundings from the API and render as labelled markers.
   *
   * @param {L.Map} map
   * @param {object} bounds
   * @param {number} n
   * @returns {Promise<L.LayerGroup>}
   */
  async function fetchDepthSoundings(map, bounds, n = 50) {
    const params = new URLSearchParams({
      bounds: `${bounds.minLon},${bounds.minLat},${bounds.maxLon},${bounds.maxLat}`,
      n: String(n),
    });

    const url = `/api/depth-soundings?${params.toString()}`;

    return new Promise((resolve, reject) => {
      fetch(url)
        .then(res => {
          if (!res.ok) throw new Error(`depth-soundings API ${res.status}`);
          return res.json();
        })
        .then(data => {
          const group = L.featureGroup();
          (data.points || []).forEach(pt => {
            const color = depthToColor(pt.depth_m);
            const marker = L.circleMarker([pt.lat, pt.lon], {
              radius: 5,
              fillColor: color,
              color: "#ffffff",
              weight: 1,
              fillOpacity: 0.9,
            });
            marker.bindTooltip(`${pt.depth_m.toFixed(1)} m`, {
              permanent: false,
              direction: "top",
              className: "iho-tooltip",
            });
            group.addLayer(marker);
          });
          resolve(group);
        })
        .catch(reject);
    });
  }

  // ── Depth at Click (interactive sounding tool) ──────────────────────────────

  /**
   * Activate "click for depth" tool on the map.
   * Shows a popup at clicked point with depth from the nearest bathy source.
   *
   * @param {L.Map} map
   * @param {Function} getDepthPromise — fn(lat, lon) → Promise{ depth_m }
   */
  function activateDepthClickTool(map, getDepthPromise) {
    map._depthClickHandler = (e) => {
      const { lat, lng } = e.latlng;
      getDepthPromise(lat, lng)
        .then(depth => {
          const color = depthToColor(depth);
          const popup = L.popup()
            .setLatLng(e.latlng)
            .setContent(`
              <div style="font-family:Inter,sans-serif;font-size:12px;text-align:center;padding:4px 8px;">
                <div style="font-size:10px;color:#94a3b8;margin-bottom:2px;">Sounding</div>
                <div style="font-size:20px;font-weight:700;color:${color};">${depth.toFixed(1)} m</div>
              </div>
            `)
            .openOn(map);
        })
        .catch(() => {});
    };
    map.on("click", map._depthClickHandler);
  }

  function deactivateDepthClickTool(map) {
    if (map._depthClickHandler) {
      map.off("click", map._depthClickHandler);
      delete map._depthClickHandler;
    }
  }

  // ── Danger Zones from Reef Candidates ───────────────────────────────────────

  /**
   * Render candidate reef areas with confidence < threshold as danger zones.
   *
   * @param {L.Map} map
   * @param {object} options — { threshold: 30, color: '#ff4444' }
   * @returns {Promise<L.GeoJSON>}
   */
  async function renderDangerZones(map, options = {}) {
    const { threshold = 30, color = "#ff4444" } = options;

    const url = "/api/candidates";
    return new Promise((resolve, reject) => {
      fetch(url)
        .then(res => res.ok ? res.json() : Promise.reject(res.status))
        .then(data => {
          const dangerous = {
            type: "FeatureCollection",
            features: (data.features || []).filter(f => {
              const score = f.properties?.confidence_score ?? f.properties?.score ?? 100;
              return score < threshold;
            }),
          };

          const layer = L.geoJSON(dangerous, {
            style: {
              color: color,
              weight: 2,
              dashArray: "4,3",
              fillColor: color,
              fillOpacity: 0.15,
            },
            onEachFeature: (feature, layer) => {
              const score = feature.properties?.confidence_score ?? feature.properties?.score;
              if (score !== undefined) {
                layer.bindTooltip(
                  `⚠️ Perigo — confiaça ${score}%`,
                  { permanent: false, className: "iho-tooltip" }
                );
              }
            },
          });
          resolve(layer);
        })
        .catch(reject);
    });
  }

  // ── Export symbol library ─────────────────────────────────────────────────

  /**
   * Generate an IHO standard danger symbol SVG.
   * type: 'rock' | 'w recharted' | 'wreck' | 'obstruction'
   */
  function dangerSymbolSVG(type = "rock") {
    const symbols = {
      rock: `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24">
        <circle cx="12" cy="12" r="10" fill="#FFD700" stroke="#000" stroke-width="1.5"/>
        <text x="12" y="17" text-anchor="middle" font-size="14" font-weight="900" fill="#000" font-family="sans-serif">⛿</text>
      </svg>`,
      wreck: `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24">
        <polygon points="12,3 21,21 3,21" fill="#FFD700" stroke="#000" stroke-width="1.5"/>
        <text x="12" y="17" text-anchor="middle" font-size="10" font-weight="700" fill="#000" font-family="sans-serif">W</text>
      </svg>`,
    };
    return symbols[type] || symbols.rock;
  }

  // ── Public API ────────────────────────────────────────────────────────────────

  window.IHOChart = {
    IHO_BANDS,
    IHO_LAND_COLOR,
    depthToColor,
    depthToRgba,
    renderDepthBandsOnCanvas,
    buildDepthLegend,
    fetchIsobathLayer,
    fetchDepthSoundings,
    activateDepthClickTool,
    deactivateDepthClickTool,
    renderDangerZones,
    dangerSymbolSVG,
  };

})();
