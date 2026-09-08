"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const path = require("node:path");
const { Worker } = require("node:worker_threads");
const C = require("../docs/deblur-core.js");
function reflected(i, n) {
  while (i < 0 || i >= n) i = i < 0 ? -i : 2 * n - i - 2;
  return i;
}
function spatialBlur(image, w, h, k, ks) {
  const out = new Float32Array(w * h),
    mid = (ks - 1) / 2;
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      let sum = 0;
      for (let ky = 0; ky < ks; ky++)
        for (let kx = 0; kx < ks; kx++)
          sum +=
            k[ky * ks + kx] *
            image[reflected(y - ky + mid, h) * w + reflected(x - kx + mid, w)];
      out[y * w + x] = sum;
    }
  return out;
}
function fixture(w, h) {
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      let v = 0.3 + 0.07 * Math.sin(x * 0.24) + 0.1 * Math.cos(y * 0.18);
      if (x > w * 0.2 && x < w * 0.48 && y > h * 0.15 && y < h * 0.72)
        v += 0.38;
      if (Math.hypot(x - w * 0.73, y - h * 0.6) < h * 0.2) v += 0.32;
      out[y * w + x] = Math.max(0, Math.min(1, v));
    }
  return out;
}
function mse(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += (a[i] - b[i]) ** 2;
  return sum / a.length;
}
function rgba(gray) {
  const out = new Uint8ClampedArray(gray.length * 4);
  for (let i = 0; i < gray.length; i++) {
    out.set([gray[i] * 255, gray[i] * 255, gray[i] * 255, i % 256], i * 4);
  }
  return out;
}
test("reflection FFT matches independent spatial convolution with an asymmetric PSF", () => {
  const w = 43,
    h = 35,
    a = fixture(w, h),
    k = new Float32Array([0, 0.1, 0, 0, 0.3, 0.2, 0, 0.4, 0]);
  const expected = spatialBlur(a, w, h, k, 3),
    actual = C.blurChannel(a, w, h, k, 3);
  assert.ok(mse(expected, actual) < 1e-12);
});
test("regularization preserves constant brightness, including non-power-of-two borders", () => {
  for (const [model, amount] of [
    ["motion", 15],
    ["defocus", 3],
  ]) {
    const { k, size } = C.parametricKernel(model, amount, 27),
      source = new Float32Array(73 * 59).fill(0.65);
    const restored = C.deconvChannel(source, 73, 59, k, size, 0.03);
    assert.ok(mse(source, restored) < 1e-12);
  }
});
test("known motion and defocus recovery improves independent synthetic ground truth", () => {
  const w = 96,
    h = 80,
    sharp = fixture(w, h);
  for (const [model, amount, angle] of [
    ["motion", 9, 0],
    ["motion", 13, 32],
    ["defocus", 3, 0],
  ]) {
    const { k, size } = C.parametricKernel(model, amount, angle),
      blurred = spatialBlur(sharp, w, h, k, size);
    let seed = 19;
    for (let i = 0; i < blurred.length; i++) {
      seed = (1664525 * seed + 1013904223) >>> 0;
      blurred[i] += 0.004 * (seed / 2 ** 32 - 0.5);
    }
    const output = C.deconvChannel(blurred, w, h, k, size, 0.001);
    const gain = 10 * Math.log10(mse(blurred, sharp) / mse(output, sharp));
    console.log(
      `${model} ${amount}px ${angle}deg: ${gain.toFixed(2)} dB PSNR gain vs blurred`,
    );
    assert.ok(gain > 3, `Expected useful recovery, got ${gain} dB`);
  }
});
test("valid bright content is not rejected by an absolute clipped-pixel threshold", () => {
  const obs = new Float32Array(32 * 32 * 3).fill(1),
    { k, size } = C.parametricKernel("motion", 5);
  const q = C.quality(obs, obs, 32, 32, k, size);
  assert.equal(q.extraClip, 0);
  assert.equal(q.clipFrac, 1);
  assert.ok(q.score < 1e-6);
});
test("native pipeline retains tiny dimensions, alpha, all five finite outputs, and settings", async () => {
  const w = 13,
    h = 9,
    source = rgba(new Float32Array(w * h).fill(0.55));
  const result = await C.run({ w, h, rgba: source }, { model: "auto" });
  assert.equal(result.w, w);
  assert.equal(result.h, h);
  assert.equal(result.scale, 1);
  assert.equal(Object.keys(result.methods).length, 5);
  for (const item of Object.values(result.methods)) {
    assert.equal(item.rgba.length, source.length);
    assert.ok(Number.isFinite(item.q.score));
    for (let i = 0; i < w * h; i++)
      assert.equal(item.rgba[i * 4 + 3], source[i * 4 + 3]);
  }
});
test("overlap tiles cover each pixel and do not create seams in a constant image", async () => {
  const w = 151,
    h = 31,
    source = rgba(new Float32Array(w * h).fill(0.55));
  const result = await C.run(
    { w, h, rgba: source },
    { model: "motion", length: 5, tileSize: 128 },
  );
  assert.ok(result.tiles > 1);
  for (const item of Object.values(result.methods))
    for (let i = 0; i < w * h; i++) {
      assert.ok(Math.abs(item.rgba[i * 4] - source[i * 4]) <= 1);
      assert.equal(item.rgba[i * 4 + 3], source[i * 4 + 3]);
    }
});
test("sliding dark-channel projection matches independent immutable patch minima", () => {
  const w = 19,
    h = 17,
    source = fixture(w, h),
    expected = new Float32Array(source),
    r = 2;
  // Add unique low-valued pixels to avoid ambiguous argmin ties.
  for (let i = 0; i < source.length; i++)
    source[i] = source[i] * 0.02 + i * 1e-7;
  expected.set(source);
  for (let y = r; y < h - r; y++)
    for (let x = r; x < w - r; x++) {
      let index = y * w + x;
      for (let yy = y - r; yy <= y + r; yy++)
        for (let xx = x - r; xx <= x + r; xx++)
          if (source[yy * w + xx] < source[index]) index = yy * w + xx;
      if (source[index] ** 2 < 0.004 / 0.03) expected[index] = 0;
    }
  assert.deepEqual(
    C.localMinProjection(source, w, h, 5, 0.004, 0.03),
    expected,
  );
});
test("invalid options and oversized native jobs fail explicitly", async () => {
  assert.throws(() => C.validateOptions({ length: NaN }), /Invalid length/);
  assert.throws(() => C.validateOptions({ model: "unknown" }), /Invalid/);
  await assert.rejects(
    C.run({ w: 4000, h: 3001, rgba: { length: 4000 * 3001 * 4 } }),
    /12 megapixels/,
  );
});
test("worker protocol transfers all five outputs and supports terminating a job", async () => {
  const filename = path.resolve(__dirname, "../docs/deblur-worker.js");
  const wrapper = `const {parentPort,workerData}=require('node:worker_threads'); const path=require('node:path'); global.self=global; global.importScripts=p=>require(path.join(path.dirname(workerData),p)); global.postMessage=(data,transfer)=>parentPort.postMessage(data,transfer); require(workerData); parentPort.on('message',data=>self.onmessage({data}));`;
  const worker = new Worker(wrapper, { eval: true, workerData: filename });
  try {
    const messages = [],
      pixels = rgba(new Float32Array(12 * 9).fill(0.6));
    const result = await new Promise((resolve, reject) => {
      worker.on("error", reject);
      worker.on("message", (data) => {
        messages.push(data);
        if (data.type === "error") reject(new Error(data.message));
        if (data.type === "result") resolve(data);
      });
      worker.postMessage(
        {
          id: 7,
          image: { w: 12, h: 9, rgba: pixels },
          options: { model: "motion", length: 3 },
        },
        [pixels.buffer],
      );
    });
    assert.equal(pixels.byteLength, 0);
    assert.equal(result.id, 7);
    assert.equal(Object.keys(result.result.methods).length, 5);
    assert.ok(messages.some((message) => message.type === "progress"));
  } finally {
    await worker.terminate();
  }
  const cancelled = new Worker(wrapper, { eval: true, workerData: filename });
  try {
    await new Promise((resolve, reject) => {
      cancelled.once("error", reject);
      cancelled.once("message", resolve);
      cancelled.postMessage({
        id: 8,
        image: { w: 80, h: 64, rgba: rgba(fixture(80, 64)) },
        options: {},
      });
    });
  } finally {
    assert.equal(await cancelled.terminate(), 1);
  }
});

test("reflected forward and adjoint satisfy the inner-product identity", () => {
  const w = 31,
    h = 27,
    x = fixture(w, h),
    y = Float32Array.from(x, (v, i) => Math.cos(i * 0.17));
  const k = new Float32Array([0, 0.1, 0, 0, 0.3, 0.2, 0, 0.4, 0]),
    ax = C.blurChannel(x, w, h, k, 3),
    aty = C.adjointBlur(y, w, h, k, 3);
  let lhs = 0,
    rhs = 0;
  for (let i = 0; i < x.length; i++) {
    lhs += ax[i] * y[i];
    rhs += x[i] * aty[i];
  }
  assert.ok(Math.abs(lhs - rhs) < 1e-5);
});

test("the exported recommended restoration improves a blurred synthetic image", async () => {
  const w = 64,
    h = 48,
    sharp = fixture(w, h),
    { k, size } = C.parametricKernel("motion", 7, 25);
  const blurry = spatialBlur(sharp, w, h, k, size),
    input = rgba(blurry);
  for (let i = 3; i < input.length; i += 4) input[i] = 255;
  const result = await C.run(
    { w, h, rgba: input },
    { model: "motion", length: 7, angle: 25 },
  );
  const output = C.rgbaData(result.methods[result.recommended].rgba, w, h).gray;
  const gain = 10 * Math.log10(mse(blurry, sharp) / mse(output, sharp));
  console.log(
    `Exported ${result.recommended}: ${gain.toFixed(2)} dB gain vs blurred`,
  );
  assert.ok(gain > 1, "The full pipeline should retain useful detail recovery");
});

test('automatic estimation handles small textured images and subpixel preview blur', async () => {
  const w=24,h=27,pixels=rgba(fixture(w,h));
  const estimate=await C.estimateKernel({w,h,rgba:pixels},C.validateOptions({quality:'fast'}),()=>{});
  assert.ok(estimate.candidateCount>=1);
  assert.ok(estimate.k.every(Number.isFinite));
  assert.ok(Math.abs(estimate.k.reduce((a,b)=>a+b,0)-1)<1e-5);
  const tiny=C.parametricKernel('defocus',.1);
  assert.equal(tiny.k[(tiny.k.length-1)/2],1);
});
