// Visualize what gee/python/export_event.py would export.
//
// Paste into https://code.earthengine.google.com/ — shows:
//   * Each hotspot bbox (coloured)
//   * Tile grid (512x512 @ 10 m, same split as _tile_bbox)
//   * S1 VV / VH refined-Lee composite (same recipe as s1_preprocess.load_s1)
//   * S2 MNDWI, Otsu water, JRC permanent, final weak flood label
//     (same recipe as weak_labels.generate_weak_label)
//   * Per-tile flood-pixel count + % chart, with MIN_FLOOD_PX threshold line
//   * Which tiles would pass / fail the export filter
//
// Change EVENT_KEY below to switch events. Keep params in sync with
// configs/events.yaml and configs/events.yaml::export.

// ---------------- Event config (mirror configs/events.yaml) ----------------
// Multi-bbox events are river-corridor hotspots; single-bbox events keep the
// whole AOI and are wrapped as a 1-element list here.
var EVENTS = {
  assam2022: {
    bboxes: [
      [89.7, 25.9, 90.8, 26.3],   // h00 Dhubri–Goalpara
      [92.1, 26.1, 93.0, 26.6],   // h01 Morigaon–Nagaon
      [92.7, 26.5, 93.6, 26.9],   // h02 Kaziranga
      [93.7, 26.7, 94.5, 27.1],   // h03 Majuli–Jorhat
      [94.5, 27.2, 95.6, 27.7]    // h04 Dibrugarh–Tinsukia
    ],
    s1: ['2022-06-15', '2022-07-20'],
    s2: ['2022-06-10', '2022-07-25']
  },
  bihar2021:   {bboxes: [[83.0, 24.5, 88.5, 27.0]], s1: ['2021-07-20', '2021-08-25'], s2: ['2021-07-15', '2021-08-30']},
  kerala2018:  {bboxes: [[74.5,  8.0, 77.5, 12.5]], s1: ['2018-08-10', '2018-08-25'], s2: ['2018-08-01', '2018-08-31']},
  chennai2015: {bboxes: [[79.8, 12.5, 80.5, 13.5]], s1: ['2015-11-25', '2015-12-15'], s2: ['2015-11-15', '2015-12-31']}
};

var EVENT_KEY     = 'assam2022';
var SCALE_M       = 10;
var TILE_PX       = 512;
var ORBIT         = 'DESCENDING';
var APPLY_LEE     = true;
var MIN_FLOOD_PX  = 500;     // match configs/events.yaml::export.min_flood_px
var CHART_TOP_N   = 60;      // chart shows top-N tiles by flood_pct
var MAX_TILE_OUTLINES = 1500;// safety cap on tile outlines drawn

var cfg = EVENTS[EVENT_KEY];
var bboxes = cfg.bboxes;
var s1Start = cfg.s1[0], s1End = cfg.s1[1];
var s2Start = cfg.s2[0], s2End = cfg.s2[1];

// Envelope for S1/label load + map centering
var envelope = [
  Math.min.apply(null, bboxes.map(function(b){return b[0];})),
  Math.min.apply(null, bboxes.map(function(b){return b[1];})),
  Math.max.apply(null, bboxes.map(function(b){return b[2];})),
  Math.max.apply(null, bboxes.map(function(b){return b[3];}))
];
var envAoi = ee.Geometry.Rectangle(envelope);
Map.centerObject(envAoi, 7);

// ---------------- Hotspot outlines ----------------
var PALETTE = ['ff3333', 'ffaa00', '33cc33', '00cccc', '3366ff', 'cc33cc', '996633'];

var hotspotFeats = [];
for (var h = 0; h < bboxes.length; h++) {
  var bb = bboxes[h];
  hotspotFeats.push(ee.Feature(ee.Geometry.Rectangle(bb), {
    h: h,
    tag: 'h' + (h < 10 ? '0' + h : h),
    bbox: bb.join(',')
  }));
}
var hotspotFC = ee.FeatureCollection(hotspotFeats);

// One layer per hotspot so the legend maps colour -> tag
for (var h = 0; h < bboxes.length; h++) {
  var col = PALETTE[h % PALETTE.length];
  Map.addLayer(
    ee.FeatureCollection([hotspotFeats[h]])
      .style({color: col, fillColor: '00000000', width: 3}),
    {}, 'Hotspot h' + (h < 10 ? '0' + h : h)
  );
}

// ---------------- Tile grid (mirrors _tile_bbox for each hotspot) ----------------
var M_PER_DEG_LAT = 111320.0;
var tileSizeM     = TILE_PX * SCALE_M;

var allTiles = [];
var perHotspotCount = {};
var totalCandidate = 0;

for (var hi = 0; hi < bboxes.length; hi++) {
  var bb2 = bboxes[hi];
  var midLat = (bb2[1] + bb2[3]) / 2.0;
  var stepLat = tileSizeM / M_PER_DEG_LAT;
  var stepLon = tileSizeM / (111320.0 * Math.cos(midLat * Math.PI / 180.0));

  var nLat = Math.ceil((bb2[3] - bb2[1]) / stepLat);
  var nLon = Math.ceil((bb2[2] - bb2[0]) / stepLon);
  perHotspotCount[hi] = nLat * nLon;
  totalCandidate += nLat * nLon;

  for (var iy = 0; iy < nLat; iy++) {
    for (var ix = 0; ix < nLon; ix++) {
      var lo  = bb2[0] + ix * stepLon;
      var la  = bb2[1] + iy * stepLat;
      var hi0 = Math.min(lo + stepLon, bb2[2]);
      var hi1 = Math.min(la + stepLat, bb2[3]);
      var tag = EVENT_KEY
        + '_h' + (hi < 10 ? '0' + hi : hi)
        + '_x' + (ix < 10 ? '00' + ix : ix < 100 ? '0' + ix : ix)
        + '_y' + (iy < 10 ? '00' + iy : iy < 100 ? '0' + iy : iy);
      allTiles.push(ee.Feature(ee.Geometry.Rectangle([lo, la, hi0, hi1]), {
        tag: tag, h: hi, ix: ix, iy: iy,
        colour: PALETTE[hi % PALETTE.length]
      }));
    }
  }
}
print('Candidate tiles total:', totalCandidate);
for (var hp in perHotspotCount) {
  print('  h' + (hp < 10 ? '0' + hp : hp) + ' tiles: ' + perHotspotCount[hp]);
}

var drawnTiles = allTiles.slice(0, MAX_TILE_OUTLINES);
var tileFC = ee.FeatureCollection(drawnTiles);

// Tile outlines coloured by hotspot
Map.addLayer(
  tileFC.style({styleProperty: 'colour', fillColor: '00000000', width: 1}),
  {}, 'Tile grid (' + drawnTiles.length + ' / ' + allTiles.length + ')'
);

// ---------------- S1 composite (mirror load_s1, over envelope) ----------------
function refinedLee(image) {
  var pow = ee.Image(10.0).pow(image.divide(10.0));
  var k3  = ee.Kernel.square(1);
  var m3  = pow.reduceNeighborhood(ee.Reducer.mean(), k3);
  var v3  = pow.reduceNeighborhood(ee.Reducer.variance(), k3);
  var sw  = ee.List.repeat(ee.List.repeat(1, 7), 7);
  var sk  = ee.Kernel.fixed(7, 7, sw, 3, 3, false);
  var sm  = m3.neighborhoodToBands(sk);
  var sv  = v3.neighborhoodToBands(sk);
  var g = sm.select(1).subtract(sm.select(7)).abs();
  g = g.addBands(sm.select(6).subtract(sm.select(2)).abs());
  g = g.addBands(sm.select(3).subtract(sm.select(5)).abs());
  g = g.addBands(sm.select(0).subtract(sm.select(8)).abs());
  var maxG = g.reduce(ee.Reducer.max());
  var gm   = g.eq(maxG);
  var dv = sv.select(1).multiply(gm.select(0))
    .add(sv.select(6).multiply(gm.select(1)))
    .add(sv.select(3).multiply(gm.select(2)))
    .add(sv.select(0).multiply(gm.select(3)));
  var sigma = 0.4472;
  var sVar = dv.subtract(m3.pow(2).multiply(sigma)).divide(1.0 + sigma);
  var b    = sVar.divide(dv.max(1e-9));
  var fp   = m3.add(b.multiply(pow.subtract(m3)));
  return ee.Image(10.0).multiply(fp.max(1e-9).log10()).rename(image.bandNames());
}

var s1col = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(envAoi)
  .filterDate(s1Start, s1End)
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
  .filter(ee.Filter.eq('orbitProperties_pass', ORBIT))
  .select(['VV', 'VH'])
  .map(function(img) {
    var out = img;
    if (APPLY_LEE) {
      out = ee.Image.cat([
        refinedLee(img.select('VV')).rename('VV'),
        refinedLee(img.select('VH')).rename('VH')
      ]);
    }
    return out.clamp(-50, 1).copyProperties(img, ['system:time_start']);
  });
print('S1 scene count:', s1col.size());

var s1 = s1col.median().clip(envAoi).rename(['VV', 'VH']);
Map.addLayer(s1.select('VV'), {min:-23, max:0}, 'S1 VV (dB)');
Map.addLayer(s1.select('VH'), {min:-28, max:-5}, 'S1 VH (dB)');

// ---------------- S2 MNDWI + Otsu (mirror generate_weak_label) ----------------
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(envAoi)
  .filterDate(s2Start, s2End)
  .map(function(img) { return img.updateMask(img.select('MSK_CLDPRB').lt(40)); });
print('S2 scene count:', s2.size());

var s2c   = s2.median().clip(envAoi);
var mndwi = s2c.normalizedDifference(['B3', 'B11']).rename('MNDWI');
Map.addLayer(mndwi, {min:-0.5, max:0.7, palette:['brown','white','blue']}, 'MNDWI');

function otsu(histogram) {
  var counts = ee.Array(ee.Dictionary(histogram).get('histogram'));
  var means  = ee.Array(ee.Dictionary(histogram).get('bucketMeans'));
  var size   = means.length().get([0]);
  var total  = counts.reduce(ee.Reducer.sum(), [0]).get([0]);
  var sum_   = means.multiply(counts).reduce(ee.Reducer.sum(), [0]).get([0]);
  var meanT  = sum_.divide(total);
  var idx    = ee.List.sequence(1, size);
  var bss = idx.map(function(i) {
    i = ee.Number(i);
    var aC = counts.slice(0, 0, i).reduce(ee.Reducer.sum(), [0]).get([0]);
    var aM = means.slice(0, 0, i).multiply(counts.slice(0, 0, i))
      .reduce(ee.Reducer.sum(), [0]).get([0]).divide(aC.max(1e-9));
    var bC = ee.Number(total).subtract(aC);
    var bM = ee.Number(sum_).subtract(aM.multiply(aC)).divide(bC.max(1e-9));
    return aC.multiply(aM.subtract(meanT).pow(2))
      .add(bC.multiply(bM.subtract(meanT).pow(2)));
  });
  var maxBss = ee.List(bss).reduce(ee.Reducer.max());
  var i = ee.List(bss).indexOf(maxBss);
  return ee.Number(means.get([i]));
}

var hist = mndwi.reduceRegion({
  reducer: ee.Reducer.histogram(256, 0.01),
  geometry: envAoi, scale: 30, bestEffort: true, maxPixels: 1e10
}).get('MNDWI');
var thr = otsu(hist);
print('Otsu threshold (MNDWI):', thr);

var water = mndwi.gt(ee.Image.constant(thr)).rename('water');
Map.addLayer(water.selfMask(), {palette:['cyan']}, 'Otsu water (raw)');

var jrc       = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('seasonality');
var permanent = jrc.gte(10).unmask(0);
Map.addLayer(permanent.selfMask(), {palette:['darkblue']}, 'JRC permanent');

// Final flood label exactly as export_event.py writes it
var valid = mndwi.mask();
var flood = water.and(permanent.not()).toInt16().where(valid.not(), -1).rename('label');
Map.addLayer(flood.updateMask(flood.eq(1)),  {palette:['red']},    'Weak flood label (=1)');
Map.addLayer(flood.updateMask(flood.eq(-1)), {palette:['yellow']}, 'Cloud / no-data (=-1)');

// ---------------- Per-tile flood stats (single reduceRegions call) ----------------
// Heavy op — EE may take ~30-60s to return on ~1000 tiles. Worth it to see
// exactly which tiles pass the export filter.
var floodPos = flood.eq(1).rename('flood');
var valid1   = flood.neq(-1).rename('valid');
var statsFC  = floodPos.addBands(valid1).reduceRegions({
  collection: ee.FeatureCollection(allTiles),
  reducer: ee.Reducer.sum(),
  scale: SCALE_M
});

// Derive flood_pct and keep/skip flag (client-unsafe, stays server-side)
var withStats = statsFC.map(function(f) {
  var fSum = ee.Number(f.get('flood')).max(0);
  var vSum = ee.Number(f.get('valid')).max(1);
  var pct  = fSum.divide(vSum).multiply(100);
  return f.set({
    flood_px: fSum,
    valid_px: vSum,
    flood_pct: pct,
    keep: fSum.gte(MIN_FLOOD_PX)
  });
});

// Kept vs skipped as separate layers
var kept    = withStats.filter(ee.Filter.eq('keep', 1))
  .style({color: '33ff66', fillColor: '00000000', width: 1.5});
var skipped = withStats.filter(ee.Filter.eq('keep', 0))
  .style({color: '444444', fillColor: '00000000', width: 1});
Map.addLayer(skipped, {}, 'Tiles SKIPPED (< ' + MIN_FLOOD_PX + ' flood px)');
Map.addLayer(kept,    {}, 'Tiles KEPT    (>= ' + MIN_FLOOD_PX + ' flood px)');

var keptCount = withStats.filter(ee.Filter.eq('keep', 1)).size();
print('Tiles that would be exported (keep=1):', keptCount);
print('Tiles skipped by min_flood_px filter  :', withStats.filter(ee.Filter.eq('keep', 0)).size());

// Top-N chart by flood_pct, descending
var topN = withStats.sort('flood_pct', false).limit(CHART_TOP_N);
var chartPct = ui.Chart.feature.byFeature(topN, 'tag', ['flood_pct'])
  .setChartType('ColumnChart')
  .setOptions({
    title: 'Top ' + CHART_TOP_N + ' tiles by flood pixel %',
    hAxis: {title: 'tile', slantedText: true, slantedTextAngle: 70},
    vAxis: {title: '% flood (of valid px)'},
    legend: {position: 'none'}
  });
print(chartPct);

var chartCnt = ui.Chart.feature.byFeature(topN, 'tag', ['flood_px'])
  .setChartType('ColumnChart')
  .setOptions({
    title: 'Top ' + CHART_TOP_N + ' tiles by flood pixel COUNT  (threshold = ' + MIN_FLOOD_PX + ')',
    hAxis: {title: 'tile', slantedText: true, slantedTextAngle: 70},
    vAxis: {title: 'flood pixels'},
    legend: {position: 'none'},
    series: {0: {color: '3366ff'}}
  });
print(chartCnt);

// ---------------- Summary ----------------
print('---- Summary ----');
print('Event:            ' + EVENT_KEY);
print('Hotspots:         ' + bboxes.length);
print('Envelope bbox:    ' + envelope);
print('S1 window:        ' + s1Start + ' .. ' + s1End + '  orbit=' + ORBIT + '  Lee=' + APPLY_LEE);
print('S2 window:        ' + s2Start + ' .. ' + s2End);
print('Candidate tiles:  ' + totalCandidate);
print('Tile size:        ' + TILE_PX + ' px @ ' + SCALE_M + ' m  (~' + tileSizeM + ' m)');
print('min_flood_px:     ' + MIN_FLOOD_PX + '  (tiles below this are skipped on export)');
print('Drive path:       gee_exports/' + EVENT_KEY + '/{S1,Label}/' + EVENT_KEY + '_h##_x###_y###_{S1,Label}.tif');
print('Tip: toggle "Tiles KEPT" vs "Tiles SKIPPED" to see the filter effect on the map.');
