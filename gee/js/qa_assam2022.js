// GEE Code Editor QA script — Assam 2022 floods.
//
// Paste into https://code.earthengine.google.com/ to visually verify the
// weak-label pipeline (S1 VV, MNDWI, Otsu water, JRC permanent, final label).
//
// Keep parameters in sync with configs/events.yaml::events.assam2022.

var bbox    = ee.Geometry.Rectangle([89.5, 24.0, 96.0, 27.5]);
var s1Start = '2022-06-15';
var s1End   = '2022-07-20';
var s2Start = '2022-06-10';
var s2End   = '2022-07-25';

Map.centerObject(bbox, 7);

// ---------------- Sentinel-1 ----------------
var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(bbox)
  .filterDate(s1Start, s1End)
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
  .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
  .select(['VV', 'VH'])
  .median()
  .clip(bbox);

Map.addLayer(s1.select('VV'), {min: -25, max: 0}, 'S1 VV (dB)');
Map.addLayer(s1.select('VH'), {min: -30, max: -5}, 'S1 VH (dB)');

// ---------------- Sentinel-2 MNDWI + Otsu ----------------
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(bbox)
  .filterDate(s2Start, s2End)
  .map(function(img) {
    return img.updateMask(img.select('MSK_CLDPRB').lt(40));
  })
  .median()
  .clip(bbox);

var mndwi = s2.normalizedDifference(['B3', 'B11']).rename('MNDWI');
Map.addLayer(mndwi, {min: -0.5, max: 0.7, palette: ['brown', 'white', 'blue']}, 'MNDWI');

// Server-side Otsu
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
  geometry: bbox,
  scale: 30,
  bestEffort: true,
  maxPixels: 1e10
}).get('MNDWI');

var thr = otsu(hist);
print('Otsu threshold', thr);

var water = mndwi.gt(ee.Image.constant(thr)).rename('water');
Map.addLayer(water.selfMask(), {palette: ['cyan']}, 'Otsu water (raw)');

// ---------------- JRC permanent water ----------------
var jrc = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('seasonality');
var permanent = jrc.gte(10).unmask(0);
Map.addLayer(permanent.selfMask(), {palette: ['darkblue']}, 'JRC permanent');

// ---------------- Final weak flood label ----------------
var flood = water.and(permanent.not()).rename('flood');
Map.addLayer(flood.selfMask(), {palette: ['red']}, 'Weak flood label');

print('Tip: toggle layers to verify flood extends beyond the Brahmaputra channel');
