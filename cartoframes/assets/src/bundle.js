var init = (function () {
  'use strict';

  function format(value) {
    if (Array.isArray(value)) {
      const [first, second] = value;
      if (first === -Infinity) {
        return `< ${formatValue(second)}`;
      }
      if (second === Infinity) {
        return `> ${formatValue(first)}`;
      }
      return `${formatValue(first)} - ${formatValue(second)}`;
    }
    return formatValue(value);
  }

  function formatValue(value) {
    if (typeof value === 'number') {
      return formatNumber(value);
    }
    return value;
  }

  function formatNumber(value) {
    const log = Math.log10(Math.abs(value));

    if ((log > 4 || log < -2.00000001) && value) {
      return value.toExponential(2);
    }
    
    if (!Number.isInteger(value)) {
      return value.toLocaleString(undefined, {
        minimumFractionDigits: 2,
        maximumFractionDigits: 3
      });
    }
    
    return value.toLocaleString();
  }

  function updateViewport(map) {
    function updateMapInfo(map) {
      const mapInfo$ = document.getElementById('map-info');
    
      const center = map.getCenter();
      const lat = center.lat.toFixed(6);
      const lng = center.lng.toFixed(6);
      const zoom = map.getZoom().toFixed(2);
    
      mapInfo$.innerText = `viewport={'zoom': ${zoom}, 'lat': ${lat}, 'lng': ${lng}}`;
    }

    map.on('zoom', updateMapInfo);
    map.on('move', updateMapInfo); 
  }

  function getBasecolorSettings(basecolor) {
    return {
      'version': 8,
      'sources': {},
      'layers': [{
          'id': 'background',
          'type': 'background',
          'paint': {
              'background-color': basecolor
          }
      }]
    };
  }

  function createDefaultLegend(layers) {
    const defaultLegendContainer = document.querySelector('#defaultLegendContainer');
    defaultLegendContainer.style.display = 'none';

    AsBridge.VL.Legends.layersLegend(
      '#defaultLegend',
      layers,
      {
        onLoad: () => defaultLegendContainer.style.display = 'unset'
      }
    );
  }

  function createLegend(layer, legendData, layerIndex, mapIndex) {
    const element = document.querySelector(`#layer${layerIndex}_map${mapIndex}_legend`);

    if (legendData.prop) {
      const config = { othersLabel: 'Others' };  // TODO: i18n
      const opts = { format, config };

      if (legendData.type.startsWith('size-continuous')) {
        config.samples = 4;
      }
      
      AsBridge.VL.Legends.rampLegend(
        element,
        layer,
        legendData.prop,
        opts
      );
    }
  }

  function renderWidget(widget, value) {
    widget.element = widget.element || document.querySelector(`#${widget.id}-value`);
    
    if (value && widget.element) {
      widget.element.innerText = typeof value === 'number' ? format(value) : value;
    }
  }

  function renderBridge(bridge, widget) {
    widget.element = widget.element || document.querySelector(`#${widget.id}`);

    switch (widget.type) {
      case 'histogram':
        bridge.histogram(widget.element, widget.value, widget.options);
        break;
      case 'category':
        bridge.category(widget.element, widget.value, widget.options);
        break;
      case 'animation':
        widget.options.propertyName = widget.prop;
        bridge.animationControls(widget.element, widget.value, widget.options);
        break;
      case 'time-series':
        widget.options.propertyName = widget.prop;
        bridge.timeSeries(widget.element, widget.value, widget.options);
        break;
    }
  }

  function bridgeLayerWidgets(carto, mapLayer, mapSource, map, widgets) {
    const bridge = new AsBridge.VL.Bridge({
      carto: carto,
      layer: mapLayer,
      source: mapSource,
      map: map
    });

    widgets
      .filter((widget) => widget.has_bridge)
      .forEach((widget) => renderBridge(bridge, widget));

    bridge.build();
  }

  /** From https://github.com/errwischt/stacktrace-parser/blob/master/src/stack-trace-parser.js */

  /**
   * This parses the different stack traces and puts them into one format
   * This borrows heavily from TraceKit (https://github.com/csnover/TraceKit)
   */

  const UNKNOWN_FUNCTION = '<unknown>';
  const chromeRe = /^\s*at (.*?) ?\(((?:file|https?|blob|chrome-extension|native|eval|webpack|<anonymous>|\/).*?)(?::(\d+))?(?::(\d+))?\)?\s*$/i;
  const chromeEvalRe = /\((\S*)(?::(\d+))(?::(\d+))\)/;
  const winjsRe = /^\s*at (?:((?:\[object object\])?.+) )?\(?((?:file|ms-appx|https?|webpack|blob):.*?):(\d+)(?::(\d+))?\)?\s*$/i;
  const geckoRe = /^\s*(.*?)(?:\((.*?)\))?(?:^|@)((?:file|https?|blob|chrome|webpack|resource|\[native).*?|[^@]*bundle)(?::(\d+))?(?::(\d+))?\s*$/i;
  const geckoEvalRe = /(\S+) line (\d+)(?: > eval line \d+)* > eval/i;

  function parse(stackString) {
    const lines = stackString.split('\n');

    return lines.reduce((stack, line) => {
      const parseResult =
        parseChrome(line) ||
        parseWinjs(line) ||
        parseGecko(line);

      if (parseResult) {
        stack.push(parseResult);
      }

      return stack;
    }, []);
  }

  function parseChrome(line) {
    const parts = chromeRe.exec(line);

    if (!parts) {
      return null;
    }

    const isNative = parts[2] && parts[2].indexOf('native') === 0; // start of line
    const isEval = parts[2] && parts[2].indexOf('eval') === 0; // start of line

    const submatch = chromeEvalRe.exec(parts[2]);
    if (isEval && submatch != null) {
      // throw out eval line/column and use top-most line/column number
      parts[2] = submatch[1]; // url
      parts[3] = submatch[2]; // line
      parts[4] = submatch[3]; // column
    }

    return {
      file: !isNative ? parts[2] : null,
      methodName: parts[1] || UNKNOWN_FUNCTION,
      arguments: isNative ? [parts[2]] : [],
      lineNumber: parts[3] ? +parts[3] : null,
      column: parts[4] ? +parts[4] : null,
    };
  }

  function parseWinjs(line) {
    const parts = winjsRe.exec(line);

    if (!parts) {
      return null;
    }

    return {
      file: parts[2],
      methodName: parts[1] || UNKNOWN_FUNCTION,
      arguments: [],
      lineNumber: +parts[3],
      column: parts[4] ? +parts[4] : null,
    };
  }

  function parseGecko(line) {
    const parts = geckoRe.exec(line);

    if (!parts) {
      return null;
    }

    const isEval = parts[3] && parts[3].indexOf(' > eval') > -1;

    const submatch = geckoEvalRe.exec(parts[3]);
    if (isEval && submatch != null) {
      // throw out eval line/column and use top-most line number
      parts[3] = submatch[1];
      parts[4] = submatch[2];
      parts[5] = null; // no column when eval
    }

    return {
      file: parts[3],
      methodName: parts[1] || UNKNOWN_FUNCTION,
      arguments: parts[2] ? parts[2].split(',') : [],
      lineNumber: parts[4] ? +parts[4] : null,
      column: parts[5] ? +parts[5] : null,
    };
  }

  function displayError(e) {
    const error$ = document.getElementById('error-container');
    const errors$ = error$.getElementsByClassName('errors');
    const stacktrace$ = document.getElementById('error-stacktrace');

    errors$[0].innerHTML = e.name;
    errors$[1].innerHTML = e.name;
    errors$[2].innerHTML = e.type;
    errors$[3].innerHTML = e.message.replace(e.type, '');

    error$.style.visibility = 'visible';

    const stack = parse(e.stack);
    const list = stack.map(item => {
      return `<li>
      at <span class="stacktrace-method">${item.methodName}:</span>
      (${item.file}:${item.lineNumber}:${item.column})
    </li>`;
    });

    stacktrace$.innerHTML = list.join('\n');
  }

  function SourceFactory() {
    const sourceTypes = { GeoJSON, Query, MVT };

    this.createSource = (layer) => {
      return sourceTypes[layer.type](layer);
    };
  }

  function GeoJSON(layer) {
    new carto.source.GeoJSON(_decodeJSONQuery(layer.query));
  }

  function Query(layer) {
    const auth = {
      username: layer.credentials.username,
      apiKey: layer.credentials.api_key || 'default_public'
    };

    const config = {
      serverURL: layer.credentials.base_url || `https://${layer.credentials.username}.carto.com/`
    };

    return new carto.source.SQL(layer.query, auth, config);
  }

  function MVT(layer) {
    return new carto.source.MVT(layer.query.file, JSON.parse(layer.query.metadata));
  }

  function _decodeJSONQuery(query) {
    return JSON.parse(Base64.decode(query.replace(/b\'/, '\'')));
  }

  function resetPopupClick(interactivity) {
    interactivity.off('featureClick');
  }

  function resetPopupHover(interactivity) {
    interactivity.off('featureHover');
  }

  function setPopupsClick(map, popup, interactivity, attrs) {
    interactivity.on('featureClick', (event) => {
      updatePopup(map, popup, event, attrs);
    });
  }

  function setPopupsHover(map, popup, interactivity, attrs) {
    interactivity.on('featureHover', (event) => {
      updatePopup(map, popup, event, attrs);
    });
  }

  function updatePopup(map, popup, event, attrs) {
    if (event.features.length > 0) {
      let popupHTML = '';
      const layerIDs = [];

      for (const feature of event.features) {
        if (layerIDs.includes(feature.layerId)) {
          continue;
        }
        // Track layers to add only one feature per layer
        layerIDs.push(feature.layerId);
    
        for (const item of attrs) {
          const variable = feature.variables[item.name];
          if (variable) {
            let value = variable.value;
            value = formatValue(value);

            popupHTML = `
            <span class="popup-name">${item.title}</span>
            <span class="popup-value">${value}</span>
          ` + popupHTML;
          }
        }
      }

      popup
          .setLngLat([event.coordinates.lng, event.coordinates.lat])
          .setHTML(`<div class="popup-content">${popupHTML}</div>`);

      if (!popup.isOpen()) {
        popup.addTo(map);
      }
    } else {
      popup.remove();
    }
  }

  function setInteractivity(map, interactiveLayers, interactiveMapLayers) {
    const interactivity = new carto.Interactivity(interactiveMapLayers);
    const popup = new mapboxgl.Popup({
      closeButton: false,
      closeOnClick: false
    });

    const { clickAttrs, hoverAttrs } = _setInteractivityAttrs(interactiveLayers);

    resetPopupClick(map);
    resetPopupHover(map);

    if (clickAttrs.length > 0) {
      setPopupsClick(map, popup, interactivity, clickAttrs);
    }

    if (hoverAttrs.length > 0) {
      setPopupsHover(map, popup, interactivity, hoverAttrs);
    }
  }

  function _setInteractivityAttrs(interactiveLayers) {
    let clickAttrs = [];
    let hoverAttrs = [];

    interactiveLayers.forEach((interactiveLayer) => {
      interactiveLayer.interactivity.forEach((interactivityDef) => {
        if (interactivityDef.event === 'click') {
          clickAttrs = clickAttrs.concat(interactivityDef.attrs);
        } else if (interactivityDef.event === 'hover') {
          hoverAttrs = hoverAttrs.concat(interactivityDef.attrs);
        }
      });
    });

    return { clickAttrs, hoverAttrs };
  }

  const BASEMAPS = {
    DarkMatter: carto.basemaps.darkmatter,
    Voyager: carto.basemaps.voyager,
    Positron: carto.basemaps.positron
  };

  const attributionControl = new mapboxgl.AttributionControl({
    compact: false
  });

  const FIT_BOUNDS_SETTINGS = { animate: false, padding: 50, maxZoom: 14 };

  async function setReady(settings) {
    try {
      if (settings.maps) {
        return await initMaps(settings.maps);
      }
      
      return await initMap(settings);
    } catch (e) {
      displayError(e);
    }
  }

  async function initMaps(maps) {
    return await maps.map(async function (mapSettings, mapIndex) {
      return await initMap(mapSettings, mapIndex);
    });
  }

  async function initMap(settings, mapIndex) {
    const basecolor = getBasecolorSettings(settings.basecolor);
    const basemapStyle =  BASEMAPS[settings.basemap] || settings.basemap || basecolor;
    const container = mapIndex !== undefined ? `map-${mapIndex}` : 'map';
    const map = createMap(container, basemapStyle, settings.bounds, settings.mapboxtoken);

    if (settings.show_info) {
      updateViewport(map);
    }

    if (settings.camera) {
      map.flyTo(settings.camera);
    }

    return await initLayers(map, settings, mapIndex);
  }

  async function initLayers(map, settings, mapIndex=0) {
    const mapLayers = [];
    const interactiveLayers = [];
    const interactiveMapLayers = [];
    const factory = new SourceFactory();

    settings.layers.forEach((layer, index) => {
      const mapSource = factory.createSource(layer);
      const mapViz = new carto.Viz(layer.viz);
      const mapLayer = new carto.Layer(`layer${index}`, mapSource, mapViz);

      mapLayers.push(mapLayer);

      try {
        mapLayer._updateLayer.catch(displayError);
      } catch (e) {
        throw e;
      }

      mapLayer.addTo(map);

      if (layer.interactivity) {
        interactiveLayers.push(layer);
        interactiveMapLayers.push(mapLayer);
      }

      if (settings.has_legends && layer.legend) {
        createLegend(mapLayer, layer.legend, settings.layers.length - index - 1, mapIndex);
      }

      if (layer.widgets.length) {
        layer.widgets.forEach((widget, widgetIndex) => {
          const id = `layer${settings.layers.length - index - 1}_widget${widgetIndex}`;
          widget.id = id;
        });

        mapLayer.on('updated', () => {
          layer.widgets
            .filter((widget) => !widget.has_bridge)
            .forEach((widget) => {
              const value = widget.variable_name && mapLayer.viz.variables[widget.variable_name] ?
                mapLayer.viz.variables[widget.variable_name].value
                : null;

              renderWidget(widget, value);
            });
        });

        bridgeLayerWidgets(carto, mapLayer, mapSource, map, layer.widgets);
      }
    });

    if (interactiveLayers.length > 0) {
      setInteractivity(map, interactiveLayers, interactiveMapLayers);
    }

    if (settings.default_legend) {
      createDefaultLegend(mapLayers);
    }

    return Promise.resolve(mapLayers);
  }

  function createMap(container, basemapStyle, bounds, accessToken) {
    if (accessToken) {
      mapboxgl.accessToken = accessToken;
    }

    return new mapboxgl.Map({
      container,
      style: basemapStyle,
      zoom: 9,
      dragRotate: false,
      attributionControl: false
    })
    .addControl(attributionControl)
    .fitBounds(bounds, FIT_BOUNDS_SETTINGS);
  }

  function init(settings) {
    setReady(settings);
  }

  return init;

}());
