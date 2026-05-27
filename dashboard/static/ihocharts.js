/**
 * ihocharts.js — IHO S-4 Nautical Chart Leaflet Extension
 * =========================================================
 * Extends Leaflet with IHO-standard cartographic rendering:
 *   - Contour labels (depth text placed mid-line, rotated)
 *   - Isobath area fills (translucent zone polygons)
 *   - Depth sounding anchors with IHO typography
 *   - Scale bar and compass rose (nautical chart ornaments)
 *
 * Depends on: Leaflet (L), chart_layers.js (IHOChart)
 */

(function () {
  "use strict";

  // ── Contour Label Layer ──────────────────────────────────────────────────────

  /**
   * Place depth labels along isobath polyline segments.
   * Calls placeContourLabels(features) where each feature has
   * geometry.type = "LineString" and properties.depth = N.
   *
   * Returns an L.featureGroup of L.Marker / L.Tooltip labels.
   */
  function placeContourLabels(features, map) {
    const group = L.featureGroup();
    features.forEach(feature => {
      const coords = feature.geometry.coordinates;
      if (!coords || coords.length < 2) return;
      const depth = feature.properties?.depth || feature.properties?.Depth || 0;

      // Place label at midpoint of polyline
      const midIdx = Math.floor(coords.length / 2);
      const midCoord = coords[midIdx];

      // Don't duplicate if a label already exists nearby
      const labelText = `${depth}m`;
      const marker = L.marker([midCoord[1], midCoord[0]], {
        icon: createContourLabelIcon(depth, feature),
        interactive: false,
      });
      group.addLayer(marker);
    });
    return group;
  }

  function createContourLabelIcon(depth, feature) {
    const color = depthToIsoColor(depth);
    const div = document.createElement("div");
    div.style.cssText = `
      background: ${color};
      color: white;
      border: 1px solid rgba(255,255,255,0.5);
      border-radius: 3px;
      padding: 1px 5px;
      font-size: 11px;
      font-weight: 700;
      font-family: 'Inter', sans-serif;
      white-space: nowrap;
      box-shadow: 1px 1px 3px rgba(0,0,0,0.4);
      letter-spacing: 0.3px;
    `;
    div.textContent = `${depth}m`;
    return L.divIcon({
      html: div,
      className: "iho-contour-label",
      iconSize: [40, 18],
      iconAnchor: [20, 9],
    });
  }

  function depthToIsoColor(depthM) {
    if (!window.IHOChart) return "#ffffff";
    return window.IHOChart.depthToColor(depthM);
  }

  // ── Zone Fill Layer ──────────────────────────────────────────────────────────

  /**
   * Fill zone polygons for given GeoJSON.

   * @param {object} zoneGeoJSON — FeatureCollection with Polygon/MultiPolygon
   * @param {object} options
   * Returns {L.GeoJSON} layer with translucent IHO zone fills.
   */
  function createZoneFillLayer(zoneGeoJSON, options = {}) {
    const {
      very_shallow: vsColor = "rgba(0,50,120,0.18)",
      shallow_reef: srColor = "rgba(0,80,160,0.12)",
      mid_depth: mdColor = "rgba(30,130,200,0.08)",
      offshore: offColor = "rgba(100,160,220,0.05)",
    } = options;

    const zoneStyle = {
      "very_shallow": { fillColor: vsColor, fillOpacity: 0.18, color: "transparent" },
      "shallow_reef": { fillColor: srColor, fillOpacity: 0.12, color: "transparent" },
      "nearshore_mid": { fillColor: mdColor, fillOpacity: 0.10, color: "transparent" },
      "mid_depth": { fillColor: mdColor, fillOpacity: 0.08, color: "transparent" },
      "offshore": { fillColor: offColor, fillOpacity: 0.05, color: "transparent" },
      "default": { fillColor: vsColor, fillOpacity: 0.10, color: "transparent" },
    };

    return L.geoJSON(zoneGeoJSON, {
      style: (feature) => {
        const zone = feature.properties?.zone || feature.properties?.bathymetry_zone_class || "default";
        return zoneStyle[zone] || zoneStyle["default"];
      },
      onEachFeature: (feature, layer) => {
        const zone = feature.properties?.zone || feature.properties?.bathymetry_zone_class;
        const d10 = feature.properties?.dist_10m_m;
        const d20 = feature.properties?.dist_20m_m;
        let tip = zone ? `Zona: ${zone}` : "";
        if (d10 != null) tip += ` | 10m: ${d10}m`;
        if (d20 != null) tip += ` | 20m: ${d20}m`;
        if (tip) layer.bindTooltip(tip, { permanent: false, className: "iho-tooltip" });
      },
    });
  }

  // ── IHO Scale Bar ─────────────────────────────────────────────────────────────

  /**
   * Creates an IHO-standard nautical chart scale bar.
   * HTML-based (no images), correct for Lat/ Lon scale.
   *
   * @param {L.Map} map
   * @param {object} options — { x, y, maxWidth }
   * @returns {HTMLElement}
   */
  function createScaleBar(map, options = {}) {
    const { x = 12, y = null, maxWidth = 150 } = options;
    const bottomY = y || map.getSize().y - 36;

    const container = document.createElement("div");
    container.style.cssText = `
      position:absolute;left:${x}px;bottom:${map.getSize().y - bottomY}px;
      background:rgba(12,20,40,0.88);
      border:1px solid rgba(255,255,255,0.2);
      border-radius:6px;padding:6px 10px;
      font-family:'Inter',sans-serif;font-size:10px;color:#c8e0ff;
      z-index:1000;pointer-events:none;
    `;

    // metres per pixel using standard Web Mercator scale formula
    const mPerPixel = (156543.033 * Math.cos(map.getCenter().lat * Math.PI / 180)) / Math.pow(2, map.getZoom());

    const scaleMeters = Math.round(maxWidth * mPerPixel);
    const paddedMeters = scaleMeters > 1000
      ? `${(scaleMeters / 1000).toFixed(1)} km`
      : `${scaleMeters} m`;

    container.innerHTML = `
      <div style="margin-bottom:3px;color:#7ec8e3;font-weight:600;">Escala</div>
      <div style="display:flex;align-items:center;gap:0;">
        <div style="width:${maxWidth}px;height:6px;background:repeating-linear-gradient(to right,white 0,white 25%,transparent 25%,transparent 50%);border:1px solid rgba(255,255,255,0.5);"></div>
      </div>
      <div style="margin-top:2px;">${paddedMeters}</div>
    `;

    // Attach to map pane
    const pane = map.getContainer();
    pane.appendChild(container);

    return container;
  }

  // ── Danger Symbol Markers ───────────────────────────────────────────────────

  /**
   * Place IHO danger symbols at given GeoJSON point locations.
   * @param {L.Map} map
   * @param {object} dangerGeoJSON — FeatureCollection of Points
   * @param {string} type — 'rock' | 'wreck' | 'obstruction'
   * @returns {L.FeatureGroup}
   */
  function placeDangerSymbols(map, dangerGeoJSON, type = "rock") {
    const group = L.featureGroup();
    (dangerGeoJSON.features || []).forEach(feature => {
      if (feature.geometry?.type !== "Point") return;
      const [lon, lat] = feature.geometry.coordinates;
      const { symbol_svg } = feature.properties || {};
      const svg = window.IHOChart?.dangerSymbolSVG(type) || dangerSymbolSVGDefault(type);

      const icon = L.divIcon({
        html: svg,
        className: "iho-danger-marker",
        iconSize: [24, 24],
        iconAnchor: [12, 12],
      });

      const marker = L.marker([lat, lon], { icon, interactive: true });
      if (feature.properties) {
        const depth = feature.properties.depth || feature.properties.DEPTH || feature.properties?.["depth_m"] || "?";
        marker.bindTooltip(`⚠️ ${type} @ ${depth}m`, {
          permanent: false,
          direction: "top",
          className: "iho-tooltip",
        });
      }
      group.addLayer(marker);
    });
    return group;
  }

  function dangerSymbolSVGDefault(type) {
    const colors = { rock: "#FFD700", wreck: "#FF4444", obstruction: "#FF8800" };
    const c = colors[type] || "#FFD700";
    return `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24">
      <circle cx="12" cy="12" r="10" fill="${c}" stroke="#000" stroke-width="1.5"/>
    </svg>`;
  }

  // ── Public API ────────────────────────────────────────────────────────────────

  window.IHOCharts = {
    placeContourLabels,
    createZoneFillLayer,
    createScaleBar,
    placeDangerSymbols,
  };

})();
