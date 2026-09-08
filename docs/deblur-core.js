"use strict";
// Shared numerical core: runs unchanged in a Web Worker and in Node quality checks.
(function (root) {
  const clamp = (v, a = 0, b = 1) => Math.max(a, Math.min(b, v));
  const odd = (v) => {
    const n = Math.max(3, Math.round(v));
    return n % 2 ? n : n + 1;
  };
  const reflect = (i, n) => {
    if (n <= 1) return 0;
    const period = 2 * n - 2;
    i = ((i % period) + period) % period;
    return i < n ? i : period - i;
  };
  const nextFrame = () => new Promise((resolve) => setTimeout(resolve, 0));
  const spectrumCache = new WeakMap();
  function resizeGray(a, w, h, nw, nh) {
    const o = new Float32Array(nw * nh);
    for (let y = 0; y < nh; y++) {
      const sy = ((y + 0.5) * h) / nh - 0.5,
        y0 = Math.floor(sy),
        fy = sy - y0;
      for (let x = 0; x < nw; x++) {
        const sx = ((x + 0.5) * w) / nw - 0.5,
          x0 = Math.floor(sx),
          fx = sx - x0,
          p00 = a[reflect(y0, h) * w + reflect(x0, w)],
          p10 = a[reflect(y0, h) * w + reflect(x0 + 1, w)],
          p01 = a[reflect(y0 + 1, h) * w + reflect(x0, w)],
          p11 = a[reflect(y0 + 1, h) * w + reflect(x0 + 1, w)];
        o[y * nw + x] =
          (p00 * (1 - fx) + p10 * fx) * (1 - fy) +
          (p01 * (1 - fx) + p11 * fx) * fy;
      }
    }
    return o;
  }
  function gradients(a, w, h) {
    const gx = new Float32Array(w * h),
      gy = new Float32Array(w * h);
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        gx[i] = a[y * w + Math.min(w - 1, x + 1)] - a[i];
        gy[i] = a[Math.min(h - 1, y + 1) * w + x] - a[i];
      }
    return { gx, gy };
  }
  function edgeEnergy(a, w, h) {
    let s = 0,
      n = 0;
    for (let y = 1; y < h - 1; y++)
      for (let x = 1; x < w - 1; x++) {
        const i = y * w + x,
          gx = (a[i + 1] - a[i - 1]) * 0.5,
          gy = (a[i + w] - a[i - w]) * 0.5;
        s += Math.hypot(gx, gy);
        n++;
      }
    return s / Math.max(1, n);
  }
  function highpass(a, w, h) {
    let s = 0,
      n = 0;
    for (let y = 1; y < h - 1; y++)
      for (let x = 1; x < w - 1; x++) {
        const i = y * w + x,
          v = 4 * a[i] - a[i - 1] - a[i + 1] - a[i - w] - a[i + w];
        s += v * v;
        n++;
      }
    return Math.sqrt(s / Math.max(1, n));
  }
  function analyzeScene(data) {
    let sum = 0,
      sum2 = 0,
      dark = 0,
      bright = 0,
      sat = 0;
    for (let i = 0; i < data.gray.length; i++) {
      const v = data.gray[i];
      sum += v;
      sum2 += v * v;
      if (v < 0.08) dark++;
      if (v > 0.92) bright++;
      const r = data.rgb[i * 3],
        g = data.rgb[i * 3 + 1],
        b = data.rgb[i * 3 + 2];
      if (Math.max(r, g, b) > 0.985) sat++;
    }
    const n = data.gray.length,
      mean = sum / n,
      contrast = Math.sqrt(Math.max(0, sum2 / n - mean * mean)),
      edge = edgeEnergy(data.gray, data.w, data.h),
      hp = highpass(data.gray, data.w, data.h),
      darkFrac = dark / n,
      brightFrac = bright / n,
      satFrac = sat / n,
      lowLight = mean < 0.32 && brightFrac < 0.05,
      highSaturation = satFrac > 0.08,
      lowContrast = contrast < 0.19;
    return {
      mean,
      contrast,
      edge,
      hp,
      darkFrac,
      brightFrac,
      satFrac,
      lowLight,
      highSaturation,
      lowContrast,
      modeHint: lowLight || highSaturation ? "gradient-first" : "dark+gradient",
    };
  }
  function fft1(re, im, inverse) {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) {
        [re[i], re[j]] = [re[j], re[i]];
        [im[i], im[j]] = [im[j], im[i]];
      }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const ang = ((2 * Math.PI) / len) * (inverse ? 1 : -1),
        wr0 = Math.cos(ang),
        wi0 = Math.sin(ang);
      for (let i = 0; i < n; i += len) {
        let wr = 1,
          wi = 0;
        for (let j = 0; j < len / 2; j++) {
          const u = i + j,
            v = u + len / 2,
            tr = re[v] * wr - im[v] * wi,
            ti = re[v] * wi + im[v] * wr;
          re[v] = re[u] - tr;
          im[v] = im[u] - ti;
          re[u] += tr;
          im[u] += ti;
          const nr = wr * wr0 - wi * wi0;
          wi = wr * wi0 + wi * wr0;
          wr = nr;
        }
      }
    }
    if (inverse)
      for (let i = 0; i < n; i++) {
        re[i] /= n;
        im[i] /= n;
      }
  }
  function nextPow2(v) {
    let n = 1;
    while (n < v) n <<= 1;
    return n;
  }
  function fft2(re, im, w, h, inverse) {
    const rr = new Float64Array(Math.max(w, h)),
      ii = new Float64Array(Math.max(w, h));
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        rr[x] = re[y * w + x];
        ii[x] = im[y * w + x];
      }
      const r = rr.subarray(0, w),
        q = ii.subarray(0, w);
      fft1(r, q, inverse);
      for (let x = 0; x < w; x++) {
        re[y * w + x] = r[x];
        im[y * w + x] = q[x];
      }
    }
    for (let x = 0; x < w; x++) {
      for (let y = 0; y < h; y++) {
        rr[y] = re[y * w + x];
        ii[y] = im[y * w + x];
      }
      const r = rr.subarray(0, h),
        q = ii.subarray(0, h);
      fft1(r, q, inverse);
      for (let y = 0; y < h; y++) {
        re[y * w + x] = r[y];
        im[y * w + x] = q[y];
      }
    }
  }
  function kernelSpectrum(k, ks, w, h) {
    const re = new Float64Array(w * h),
      im = new Float64Array(w * h),
      c = (ks - 1) >> 1;
    for (let y = 0; y < ks; y++)
      for (let x = 0; x < ks; x++) {
        const yy = (y - c + h) % h,
          xx = (x - c + w) % w;
        re[yy * w + xx] = k[y * ks + x];
      }
    fft2(re, im, w, h, false);
    return { re, im };
  }
  function fftDeconvChannel(obs, w, h, k, ks, reg, prior = null, rho = 0) {
    const layout = fftLayout(w, h, ks),
      { pw, ph } = layout;
    const Y = transformReflected(obs, w, h, layout);
    const P = prior ? transformReflected(prior, w, h, layout) : null;
    const H = transferFunction(k, ks, pw, ph);
    for (let i = 0; i < Y.re.length; i++) {
      const hr = H.re[i],
        hi = H.im[i],
        yr = Y.re[i],
        yi = Y.im[i];
      const den = Math.max(
        1e-12,
        hr * hr + hi * hi + reg * H.penalty[i] + (P ? rho : 0),
      );
      Y.re[i] = (hr * yr + hi * yi + (P ? rho * P.re[i] : 0)) / den;
      Y.im[i] = (hr * yi - hi * yr + (P ? rho * P.im[i] : 0)) / den;
    }
    fft2(Y.re, Y.im, pw, ph, true);
    return cropTransform(Y.re, w, h, layout);
  }
  function blurChannel(obs, w, h, k, ks) {
    const layout = fftLayout(w, h, ks),
      { pw, ph } = layout;
    const Y = transformReflected(obs, w, h, layout),
      H = transferFunction(k, ks, pw, ph);
    for (let i = 0; i < Y.re.length; i++) {
      const r = Y.re[i],
        q = Y.im[i];
      Y.re[i] = r * H.re[i] - q * H.im[i];
      Y.im[i] = r * H.im[i] + q * H.re[i];
    }
    fft2(Y.re, Y.im, pw, ph, true);
    return cropTransform(Y.re, w, h, layout);
  }
  function makeInitKernel(size) {
    const k = new Float32Array(size * size),
      c = (size - 1) >> 1;
    k[(c - 1) * size + c] = 0.5;
    k[c * size + c] = 0.5;
    return k;
  }
  function normalizeKernel(k) {
    let s = 0;
    for (let i = 0; i < k.length; i++) {
      k[i] = Math.max(0, k[i]);
      s += k[i];
    }
    if (s <= 1e-12) return makeInitKernel(Math.round(Math.sqrt(k.length)));
    for (let i = 0; i < k.length; i++) k[i] /= s;
    return k;
  }
  function centerKernel(k, size) {
    let sx = 0,
      sy = 0,
      s = 0;
    for (let y = 0; y < size; y++)
      for (let x = 0; x < size; x++) {
        const v = k[y * size + x];
        sx += x * v;
        sy += y * v;
        s += v;
      }
    if (!s) return k;
    const c = (size - 1) / 2,
      dx = Math.round(c - sx / s),
      dy = Math.round(c - sy / s),
      o = new Float32Array(k.length);
    for (let y = 0; y < size; y++)
      for (let x = 0; x < size; x++) {
        const nx = x + dx,
          ny = y + dy;
        if (nx >= 0 && nx < size && ny >= 0 && ny < size)
          o[ny * size + nx] += k[y * size + x];
      }
    return normalizeKernel(o);
  }
  function resizeKernel(k, oldSize, newSize) {
    if (oldSize === newSize) return new Float32Array(k);
    const o = new Float32Array(newSize * newSize);
    for (let y = 0; y < newSize; y++) {
      const sy = ((y + 0.5) * oldSize) / newSize - 0.5,
        y0 = Math.floor(sy),
        fy = sy - y0;
      for (let x = 0; x < newSize; x++) {
        const sx = ((x + 0.5) * oldSize) / newSize - 0.5,
          x0 = Math.floor(sx),
          fx = sx - x0;
        let v = 0;
        for (let yy = 0; yy < 2; yy++)
          for (let xx = 0; xx < 2; xx++) {
            const ox = x0 + xx,
              oy = y0 + yy;
            if (ox >= 0 && ox < oldSize && oy >= 0 && oy < oldSize)
              v +=
                k[oy * oldSize + ox] * (xx ? fx : 1 - fx) * (yy ? fy : 1 - fy);
          }
        o[y * newSize + x] = v;
      }
    }
    return normalizeKernel(o);
  }
  function refineKernel(k, size, aggressive = false) {
    let m = 0;
    for (const v of k) m = Math.max(m, v);
    const thr = m * (aggressive ? 0.025 : 0.015),
      o = new Float32Array(k.length);
    for (let i = 0; i < k.length; i++) if (k[i] >= thr) o[i] = k[i];
    normalizeKernel(o);
    const c = (size - 1) / 2;
    let sxx = 0,
      syy = 0,
      sxy = 0;
    for (let y = 0; y < size; y++)
      for (let x = 0; x < size; x++) {
        const v = o[y * size + x],
          dx = x - c,
          dy = y - c;
        sxx += v * dx * dx;
        syy += v * dy * dy;
        sxy += v * dx * dy;
      }
    const angle = 0.5 * Math.atan2(2 * sxy, sxx - syy),
      ca = Math.cos(angle),
      sa = Math.sin(angle);
    if (aggressive)
      for (let y = 0; y < size; y++)
        for (let x = 0; x < size; x++) {
          const i = y * size + x,
            v = o[i],
            dx = x - c,
            dy = y - c,
            along = Math.abs(dx * ca + dy * sa),
            off = Math.abs(-dx * sa + dy * ca);
          if (
            v &&
            off > Math.max(3, size * 0.11) &&
            along > size * 0.2 &&
            v < m * 0.08
          )
            o[i] = 0;
        }
    return centerKernel(normalizeKernel(o), size);
  }
  function kernelStats(k, size) {
    let max = 0,
      active = 0,
      core = 0,
      sxx = 0,
      syy = 0,
      sxy = 0,
      off = 0,
      c = (size - 1) / 2;
    for (const v of k) max = Math.max(max, v);
    for (let y = 0; y < size; y++)
      for (let x = 0; x < size; x++) {
        const v = k[y * size + x];
        if (v > max * 0.03) active++;
        const dx = x - c,
          dy = y - c;
        if (Math.hypot(dx, dy) < size * 0.22) core += v;
        sxx += v * dx * dx;
        syy += v * dy * dy;
        sxy += v * dx * dy;
      }
    const angle = 0.5 * Math.atan2(2 * sxy, sxx - syy),
      ca = Math.cos(angle),
      sa = Math.sin(angle);
    for (let y = 0; y < size; y++)
      for (let x = 0; x < size; x++) {
        const v = k[y * size + x],
          dx = x - c,
          dy = y - c;
        if (Math.abs(-dx * sa + dy * ca) > Math.max(2, size * 0.13)) off += v;
      }
    const tr = sxx + syy,
      disc = Math.sqrt(Math.max(0, (sxx - syy) ** 2 + 4 * sxy * sxy)),
      l1 = (tr + disc) / 2,
      l2 = (tr - disc) / 2;
    return {
      active,
      core,
      off,
      anis: l1 > 1e-9 ? 1 - l2 / l1 : 0,
      angle: (angle * 180) / Math.PI,
    };
  }
  function thresholdGrad(gx, gy, keep = 0.08) {
    const a = [];
    for (let i = 0; i < gx.length; i++) {
      const v = gx[i] * gx[i] + gy[i] * gy[i];
      if (v > 0) a.push(v);
    }
    a.sort((x, y) => y - x);
    const t =
        a[Math.min(a.length - 1, Math.max(0, Math.floor(a.length * keep)))] ||
        0,
      ox = new Float32Array(gx),
      oy = new Float32Array(gy);
    for (let i = 0; i < ox.length; i++)
      if (ox[i] * ox[i] + oy[i] * oy[i] < t) {
        ox[i] = 0;
        oy[i] = 0;
      }
    return { gx: ox, gy: oy };
  }
  function localMinProjection(src, w, h, patch, lambda, beta) {
    // Separable sliding-window argmin, O(w*h), with immutable window values.
    // The old patch scan was O(w*h*patch^2) and changed later windows in-place.
    const r = Math.min((patch - 1) >> 1, (Math.min(w, h) - 1) >> 1);
    const horizontal = new Int32Array(src.length),
      out = new Float32Array(src);
    const queue = new Int32Array(Math.max(w, h));
    for (let y = 0; y < h; y++) {
      let head = 0,
        tail = 0;
      for (let x = 0; x < w; x++) {
        while (head < tail && src[y * w + queue[tail - 1]] >= src[y * w + x])
          tail--;
        queue[tail++] = x;
        while (head < tail && queue[head] < x - 2 * r) head++;
        if (x >= 2 * r) horizontal[y * w + x - r] = y * w + queue[head];
      }
    }
    const threshold = lambda / Math.max(beta, 1e-8);
    for (let x = r; x < w - r; x++) {
      let head = 0,
        tail = 0;
      for (let y = 0; y < h; y++) {
        while (
          head < tail &&
          src[horizontal[queue[tail - 1] * w + x]] >= src[horizontal[y * w + x]]
        )
          tail--;
        queue[tail++] = y;
        while (head < tail && queue[head] < y - 2 * r) head++;
        if (y >= 2 * r) {
          const i = horizontal[queue[head] * w + x];
          if (src[i] * src[i] < threshold) out[i] = 0;
        }
      }
    }
    return out;
  }
  function latentStep(
    src,
    w,
    h,
    k,
    ks,
    useDark,
    lambdaDark,
    lambdaGrad,
    patch,
  ) {
    let s = fftDeconvChannel(src, w, h, k, ks, Math.max(lambdaGrad, 0.00055));
    if (useDark && lambdaDark > 0) {
      let beta = Math.max(0.03, lambdaDark / 0.03);
      for (let t = 0; t < 3; t++) {
        const u = localMinProjection(s, w, h, patch, lambdaDark, beta);
        s = fftDeconvChannel(
          src,
          w,
          h,
          k,
          ks,
          Math.max(lambdaGrad, 0.00055),
          u,
          beta,
        );
        beta *= 2;
      }
    }
    return s;
  }
  function estimatePsfFromGradients(bx, by, lx, ly, w, h, ks) {
    const pw = nextPow2(w),
      ph = nextPow2(h);
    function trans(a) {
      const r = new Float64Array(pw * ph),
        i = new Float64Array(pw * ph);
      for (let y = 0; y < h; y++)
        for (let x = 0; x < w; x++) r[y * pw + x] = a[y * w + x];
      fft2(r, i, pw, ph, false);
      return { r, i };
    }
    const Bx = trans(bx),
      By = trans(by),
      Lx = trans(lx),
      Ly = trans(ly),
      R = new Float64Array(pw * ph),
      I = new Float64Array(pw * ph);
    for (let q = 0; q < R.length; q++) {
      const lxr = Lx.r[q],
        lxi = Lx.i[q],
        lyr = Ly.r[q],
        lyi = Ly.i[q],
        br = Bx.r[q],
        bi = Bx.i[q],
        cr = By.r[q],
        ci = By.i[q],
        nr = lxr * br + lxi * bi + lyr * cr + lyi * ci,
        ni = lxr * bi - lxi * br + lyr * ci - lyi * cr,
        den = lxr * lxr + lxi * lxi + lyr * lyr + lyi * lyi + 2;
      R[q] = nr / den;
      I[q] = ni / den;
    }
    fft2(R, I, pw, ph, true);
    const k = new Float32Array(ks * ks),
      c = (ks - 1) >> 1;
    for (let y = 0; y < ks; y++)
      for (let x = 0; x < ks; x++) {
        const yy = (y - c + ph) % ph,
          xx = (x - c + pw) % pw;
        k[y * ks + x] = Math.max(0, R[yy * pw + xx]);
      }
    return refineKernel(k, ks, false);
  }
  function scoreKernel(gray, w, h, k, ks, fullSupport) {
    const rest = fftDeconvChannel(gray, w, h, k, ks, 0.0015),
      rb = blurChannel(rest, w, h, k, ks);
    const border = Math.min(ks, Math.max(1, (Math.min(w, h) - 8) >> 1));
    let ss = 0,
      n = 0,
      outside = 0;
    for (let y = border; y < h - border; y++)
      for (let x = border; x < w - border; x++) {
        const i = y * w + x,
          d = rb[i] - gray[i];
        ss += d * d;
        n++;
        outside += Math.max(0, -rest[i]) + Math.max(0, rest[i] - 1);
      }
    const rm = Math.sqrt(ss / Math.max(1, n)),
      st = kernelStats(k, ks);
    const er =
      (edgeEnergy(rest, w, h) + 0.001) / (edgeEnergy(gray, w, h) + 0.001);
    const hp = (highpass(rest, w, h) + 0.003) / (highpass(gray, w, h) + 0.003);
    const artifact =
      Math.max(0, er - 4) * 0.02 +
      Math.max(0, hp - 5) * 0.015 +
      outside / Math.max(1, n);
    return {
      value: rm + artifact + Math.max(0, (fullSupport - 85) / 50) * 0.001,
      rm,
      er,
      hp,
      st,
    };
  }
  async function blindCandidate(base, fullSupport, mode, gamma, iterations) {
    const scaledSupport = odd(
        Math.max(3, Math.min(125, Math.round(fullSupport * base.scale))),
      ),
      gbase = new Float32Array(base.gray.length);
    for (let i = 0; i < gbase.length; i++)
      gbase[i] = Math.pow(clamp(base.gray[i]), gamma);
    const ratio = Math.SQRT1_2,
      maxLevels = iterations >= 5 ? 5 : 3,
      scaleLevels = [1];
    let sc = 1;
    while (scaleLevels.length < maxLevels && scaledSupport * sc > 9) {
      sc *= ratio;
      scaleLevels.unshift(sc);
    }
    let k = null,
      ks = 0,
      lambdaD = mode === "dark" ? 0.004 : 0,
      lambdaG = 0.004;
    for (const level of scaleLevels) {
      const w = Math.max(1, Math.round(base.w * level)),
        h = Math.max(1, Math.round(base.h * level)),
        y = resizeGray(gbase, base.w, base.h, w, h),
        target = odd(Math.max(5, Math.round(scaledSupport * level)));
      k = k ? resizeKernel(k, ks, target) : makeInitKernel(target);
      ks = target;
      const bg = gradients(y, w, h);
      for (let it = 0; it < iterations; it++) {
        const patch = odd(Math.max(9, Math.min(35, Math.round(35 * level)))),
          latent = latentStep(
            y,
            w,
            h,
            k,
            ks,
            mode === "dark",
            lambdaD,
            lambdaG,
            patch,
          ),
          lg = gradients(latent, w, h),
          tg = thresholdGrad(lg.gx, lg.gy, 0.08);
        k = estimatePsfFromGradients(bg.gx, bg.gy, tg.gx, tg.gy, w, h, ks);
        lambdaD = lambdaD ? Math.max(0.0001, lambdaD / 1.1) : 0;
        lambdaG = Math.max(0.0001, lambdaG / 1.1);
        await nextFrame();
      }
      k = centerKernel(k, ks);
      await nextFrame();
    }
    k = resizeKernel(k, ks, scaledSupport);
    k = refineKernel(k, scaledSupport, true);
    return {
      k,
      size: scaledSupport,
      fullSupport,
      mode,
      gamma,
      score: scoreKernel(
        base.gray,
        base.w,
        base.h,
        k,
        scaledSupport,
        fullSupport,
      ),
    };
  }
  function rgbToGray(rgb) {
    const g = new Float32Array(rgb.length / 3);
    for (let i = 0; i < g.length; i++)
      g[i] =
        0.2989360213 * rgb[i * 3] +
        0.5870430745 * rgb[i * 3 + 1] +
        0.1140209043 * rgb[i * 3 + 2];
    return g;
  }
  function restoreRGB(rgb, w, h, k, ks, reg, prior = null, rho = 0) {
    const out = new Float32Array(rgb.length),
      obs = new Float32Array(w * h),
      pr = new Float32Array(w * h);
    for (let c = 0; c < 3; c++) {
      for (let i = 0; i < obs.length; i++) {
        obs[i] = rgb[i * 3 + c];
        if (prior) pr[i] = prior[i * 3 + c];
      }
      const r = deconvChannel(obs, w, h, k, ks, reg, prior ? pr : null, rho);
      for (let i = 0; i < obs.length; i++) out[i * 3 + c] = clamp(r[i]);
    }
    return out;
  }
  function gaussianRGB(rgb, w, h) {
    const a = [0.0625, 0.25, 0.375, 0.25, 0.0625],
      tmp = new Float32Array(rgb.length),
      out = new Float32Array(rgb.length);
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++)
        for (let c = 0; c < 3; c++) {
          let v = 0;
          for (let j = -2; j <= 2; j++)
            v += rgb[(y * w + reflect(x + j, w)) * 3 + c] * a[j + 2];
          tmp[(y * w + x) * 3 + c] = v;
        }
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++)
        for (let c = 0; c < 3; c++) {
          let v = 0;
          for (let j = -2; j <= 2; j++)
            v += tmp[(reflect(y + j, h) * w + x) * 3 + c] * a[j + 2];
          out[(y * w + x) * 3 + c] = v;
        }
    return out;
  }
  function quality(obs, cand, w, h, k, ks) {
    const og = rgbToGray(obs),
      cg = rgbToGray(cand),
      rb = blurChannel(cg, w, h, k, ks);
    const border = Math.min(
      (ks - 1) >> 1,
      Math.max(0, (Math.min(w, h) - 8) >> 1),
    );
    let ss = 0,
      count = 0,
      clip = 0,
      inputClip = 0;
    for (let y = border; y < h - border; y++)
      for (let x = border; x < w - border; x++) {
        const i = y * w + x,
          d = rb[i] - og[i];
        ss += d * d;
        count++;
      }
    for (let i = 0; i < cand.length; i += 3) {
      if ([0, 1, 2].some((c) => cand[i + c] < 0.002 || cand[i + c] > 0.998))
        clip++;
      if ([0, 1, 2].some((c) => obs[i + c] < 0.002 || obs[i + c] > 0.998))
        inputClip++;
    }
    const pixels = w * h,
      rm = Math.sqrt(ss / Math.max(1, count));
    const edge =
      (edgeEnergy(cg, w, h) + 0.001) / (edgeEnergy(og, w, h) + 0.001);
    const hp = (highpass(cg, w, h) + 0.003) / (highpass(og, w, h) + 0.003);
    const clipFrac = clip / pixels,
      extraClip = Math.max(0, (clip - inputClip) / pixels);
    const penalty =
      Math.max(0, edge - 4) * 0.02 +
      Math.max(0, hp - 5) * 0.015 +
      extraClip * 0.08;
    return { score: rm + penalty, rm, edge, hp, clipFrac, extraClip };
  }
  function chooseBaseline(data, k, ks, analysis, fixedReg = null) {
    const sigma = noiseEstimate(data.gray, data.w, data.h);
    const baseReg = fixedReg ?? clamp(0.0006 + sigma * sigma * 8, 0.0006, 0.04);
    const regs =
      fixedReg === null ? [0.5, 1, 2, 4].map((v) => baseReg * v) : [fixedReg];
    let best = null;
    for (const reg of regs) {
      const raw = restoreRGB(data.rgb, data.w, data.h, k, ks, reg);
      const guarded = safeBlend(data.rgb, raw, data.w, data.h, k, ks);
      if (!best || guarded.q.score < best.q.score)
        best = { ...guarded, reg, noise: sigma };
    }
    return best;
  }
  function pnpRefine(obs, base, w, h, k, ks, reg) {
    let x = new Float32Array(base);
    const sigma = noiseEstimate(rgbToGray(obs), w, h);
    for (let t = 0; t < 3; t++) {
      const prior = bilateralRGB(
        x,
        w,
        h,
        Math.max(0.012, sigma * (3 - 0.7 * t)),
      );
      x = restoreRGB(obs, w, h, k, ks, reg, prior, 0.025 + 0.025 * t);
    }
    return x;
  }
  function extremaRefine(obs, base, w, h, k, ks, reg) {
    let x = new Float32Array(base);
    for (let t = 0; t < 3; t++) {
      const g = rgbToGray(x),
        sm = gaussianRGB(x, w, h),
        prior = new Float32Array(x.length);
      for (let y = 0; y < h; y++)
        for (let xx = 0; xx < w; xx++) {
          let mn = 1,
            mx = 0;
          for (let yy = Math.max(0, y - 2); yy <= Math.min(h - 1, y + 2); yy++)
            for (
              let x2 = Math.max(0, xx - 2);
              x2 <= Math.min(w - 1, xx + 2);
              x2++
            ) {
              const v = g[yy * w + x2];
              mn = Math.min(mn, v);
              mx = Math.max(mx, v);
            }
          const i = y * w + xx,
            dw = clamp((0.1 - mn) / 0.1),
            bw = clamp((mx - 0.9) / 0.1),
            gain = 0.035 + 0.055 * Math.max(dw, bw);
          for (let c = 0; c < 3; c++) {
            const j = i * 3 + c;
            prior[j] = clamp(x[j] + gain * (x[j] - sm[j]));
          }
        }
      x = restoreRGB(obs, w, h, k, ks, reg, prior, 0.11 + 0.05 * t);
    }
    return x;
  }
  function safeBlend(obs, cand, w, h, k, ks, anchor = obs) {
    const anchorQ = quality(obs, anchor, w, h, k, ks);
    let best = { img: anchor, q: anchorQ };
    for (const amount of [1, 0.75, 0.5, 0.25]) {
      const img =
        amount === 1
          ? cand
          : Float32Array.from(cand, (v, i) =>
              clamp(anchor[i] + amount * (v - anchor[i])),
            );
      const q = quality(obs, img, w, h, k, ks);
      if (
        q.edge <= Math.max(4.5, anchorQ.edge * 1.2) &&
        q.hp <= Math.max(5.5, anchorQ.hp * 1.2) &&
        q.extraClip <= Math.max(0.06, anchorQ.extraClip + 0.01) &&
        q.score < best.q.score
      )
        best = { img, q };
      // Accept the full candidate immediately when it passes the guard.
      if (amount === 1 && best.img === cand) break;
    }
    return best;
  }
  function motionConstrainKernel(k, size) {
    const st = kernelStats(k, size),
      angle = (st.angle * Math.PI) / 180,
      ca = Math.cos(angle),
      sa = Math.sin(angle),
      c = (size - 1) / 2,
      corridor = Math.max(1.5, size * (st.anis > 0.65 ? 0.055 : 0.085)),
      out = new Float32Array(k.length);
    let peak = 0;
    for (const v of k) peak = Math.max(peak, v);
    for (let y = 0; y < size; y++)
      for (let x = 0; x < size; x++) {
        const i = y * size + x,
          dx = x - c,
          dy = y - c,
          off = Math.abs(-dx * sa + dy * ca),
          along = Math.abs(dx * ca + dy * sa),
          keep = off <= corridor || k[i] >= peak * 0.28 || along < size * 0.08;
        if (keep) out[i] = k[i];
      }
    return refineKernel(centerKernel(normalizeKernel(out), size), size, false);
  }
  function smoothGray(src, w, h) {
    const tmp = new Float32Array(src.length),
      out = new Float32Array(src.length),
      a = [0.25, 0.5, 0.25];
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        let v = 0;
        for (let j = -1; j <= 1; j++)
          v += src[y * w + reflect(x + j, w)] * a[j + 1];
        tmp[y * w + x] = v;
      }
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        let v = 0;
        for (let j = -1; j <= 1; j++)
          v += tmp[reflect(y + j, h) * w + x] * a[j + 1];
        out[y * w + x] = v;
      }
    return out;
  }
  function localGradientMap(gray, w, h) {
    const out = new Float32Array(gray.length);
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        const i = y * w + x,
          gx =
            gray[y * w + Math.min(w - 1, x + 1)] -
            gray[y * w + Math.max(0, x - 1)],
          gy =
            gray[Math.min(h - 1, y + 1) * w + x] -
            gray[Math.max(0, y - 1) * w + x];
        out[i] = Math.hypot(gx, gy);
      }
    return out;
  }
  function localHighpassMap(gray, w, h) {
    const out = new Float32Array(gray.length);
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        const i = y * w + x,
          l = gray[y * w + Math.max(0, x - 1)],
          r = gray[y * w + Math.min(w - 1, x + 1)],
          u = gray[Math.max(0, y - 1) * w + x],
          d = gray[Math.min(h - 1, y + 1) * w + x];
        out[i] = Math.abs(4 * gray[i] - l - r - u - d);
      }
    return out;
  }
  function rgacRefine(obs, candidates, w, h, k, ks) {
    const entries = candidates.map((x) => ({ img: x.img, q: x.q })),
      observedGray = rgbToGray(obs),
      observedEdge = smoothGray(localGradientMap(observedGray, w, h), w, h),
      observedHp = smoothGray(localHighpassMap(observedGray, w, h), w, h),
      energies = [],
      globalMin = Math.min(...entries.map((x) => x.q.score)),
      globalScale = Math.max(Math.abs(globalMin), 0.01);
    for (const entry of entries) {
      const gray = rgbToGray(entry.img),
        reblur = blurChannel(gray, w, h, k, ks),
        residual = new Float32Array(gray.length),
        edge = smoothGray(localGradientMap(gray, w, h), w, h),
        hp = smoothGray(localHighpassMap(gray, w, h), w, h),
        energy = new Float32Array(gray.length),
        globalEnergy = clamp((entry.q.score - globalMin) / globalScale, 0, 4);
      for (let i = 0; i < gray.length; i++) {
        residual[i] = Math.abs(reblur[i] - observedGray[i]);
      }
      const smoothResidual = smoothGray(residual, w, h);
      for (let i = 0; i < gray.length; i++) {
        const edgeRatio = (edge[i] + 0.004) / (observedEdge[i] + 0.004),
          hpRatio = (hp[i] + 0.003) / (observedHp[i] + 0.003),
          edgePenalty = clamp((edgeRatio - 1.5) / 1.25, 0, 3),
          hpPenalty = clamp((hpRatio - 1.65) / 1.35, 0, 3),
          clipPenalty =
            entry.img[i * 3] <= 0.003 ||
            entry.img[i * 3] >= 0.997 ||
            entry.img[i * 3 + 1] <= 0.003 ||
            entry.img[i * 3 + 1] >= 0.997 ||
            entry.img[i * 3 + 2] <= 0.003 ||
            entry.img[i * 3 + 2] >= 0.997
              ? 0.35
              : 0;
        energy[i] =
          (1.75 * smoothResidual[i]) /
            Math.max(0.004, smoothResidual[i] + 0.012) +
          0.8 * edgePenalty +
          hpPenalty +
          0.7 * clipPenalty +
          0.22 * globalEnergy;
      }
      energies.push(energy);
    }
    const weights = energies.map(() => new Float32Array(w * h)),
      fused = new Float32Array(obs.length);
    for (let i = 0; i < w * h; i++) {
      let minE = Infinity;
      for (const e of energies) minE = Math.min(minE, e[i]);
      let sum = 0;
      for (let j = 0; j < energies.length; j++) {
        const weight = Math.exp(-(energies[j][i] - minE) / 0.5);
        weights[j][i] = weight;
        sum += weight;
      }
      sum = Math.max(sum, 1e-8);
      for (let j = 0; j < weights.length; j++) weights[j][i] /= sum;
      for (let c = 0; c < 3; c++) {
        let v = 0;
        for (let j = 0; j < entries.length; j++)
          v += weights[j][i] * entries[j].img[i * 3 + c];
        fused[i * 3 + c] = clamp(v);
      }
    }
    const prior = gaussianRGB(fused, w, h),
      projected = restoreRGB(obs, w, h, k, ks, 0.0012, prior, 0.12),
      guarded = safeBlend(obs, projected, w, h, k, ks),
      meanWeights = weights.map((a) => {
        let sum = 0;
        for (const v of a) sum += v;
        return sum / a.length;
      });
    return { ...guarded, weights: meanWeights };
  }
  function fftLayout(w, h, ks) {
    const halo = Math.max(32, 2 * ks);
    const pw = nextPow2(w + 2 * halo),
      ph = nextPow2(h + 2 * halo);
    return {
      padX: Math.floor((pw - w) / 2),
      padY: Math.floor((ph - h) / 2),
      pw,
      ph,
      ks,
    };
  }

  function transformReflected(src, w, h, layout) {
    const { padX, padY, pw, ph, ks } = layout;
    const re = new Float64Array(pw * ph),
      im = new Float64Array(pw * ph);
    let mean = 0;
    for (const v of src) mean += v;
    mean /= src.length;
    const ramp = (i, length, margin) =>
      Math.sin(
        Math.PI *
          0.5 *
          clamp(Math.min(i + 0.5, length - i - 0.5) / Math.max(1, margin)),
      ) ** 2;
    for (let y = 0; y < ph; y++) {
      const row = reflect(y - padY, h) * w,
        wy = ramp(y, ph, padY - ks);
      for (let x = 0; x < pw; x++) {
        const weight = wy * ramp(x, pw, padX - ks);
        re[y * pw + x] =
          mean + weight * (src[row + reflect(x - padX, w)] - mean);
      }
    }
    fft2(re, im, pw, ph, false);
    return { re, im };
  }

  function transferFunction(k, ks, pw, ph) {
    let entries = spectrumCache.get(k);
    if (!entries) {
      entries = new Map();
      spectrumCache.set(k, entries);
    }
    const key = `${pw}x${ph}`;
    if (!entries.has(key)) {
      const H = kernelSpectrum(k, ks, pw, ph),
        penalty = new Float32Array(pw * ph);
      // First-difference Tikhonov prior: exactly zero at DC, preserving brightness.
      for (let y = 0; y < ph; y++)
        for (let x = 0; x < pw; x++) {
          penalty[y * pw + x] =
            4 *
            (Math.sin((Math.PI * x) / pw) ** 2 +
              Math.sin((Math.PI * y) / ph) ** 2);
        }
      // Avoid retaining all tile shapes for a large image.
      if (entries.size >= 3) entries.delete(entries.keys().next().value);
      entries.set(key, { ...H, penalty });
    }
    return entries.get(key);
  }

  function cropTransform(re, w, h, layout) {
    const out = new Float32Array(w * h);
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++)
        out[y * w + x] = re[(y + layout.padY) * layout.pw + x + layout.padX];
    return out;
  }

  function noiseEstimate(gray, w, h) {
    const values = [],
      stride = Math.max(1, Math.floor(Math.sqrt((w * h) / 20000)));
    for (let y = 1; y < h - 1; y += stride)
      for (let x = 1; x < w - 1; x += stride) {
        const i = y * w + x;
        values.push(
          Math.abs(
            4 * gray[i] -
              2 * (gray[i - 1] + gray[i + 1] + gray[i - w] + gray[i + w]) +
              gray[i - w - 1] +
              gray[i - w + 1] +
              gray[i + w - 1] +
              gray[i + w + 1],
          ),
        );
      }
    values.sort((a, b) => a - b);
    return clamp(
      (values[values.length >> 1] || 0) / (6 * 0.67448975),
      0.0005,
      0.08,
    );
  }

  function bilateralRGB(rgb, w, h, sigma) {
    const out = new Float32Array(rgb.length),
      gray = rgbToGray(rgb);
    const variance = 2 * Math.max(0.012, sigma) ** 2;
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        let sum = 0,
          red = 0,
          green = 0,
          blue = 0;
        for (let dy = -2; dy <= 2; dy++)
          for (let dx = -2; dx <= 2; dx++) {
            const j = reflect(y + dy, h) * w + reflect(x + dx, w);
            const weight = Math.exp(
              -(dx * dx + dy * dy) / 4 - (gray[j] - gray[i]) ** 2 / variance,
            );
            sum += weight;
            red += weight * rgb[j * 3];
            green += weight * rgb[j * 3 + 1];
            blue += weight * rgb[j * 3 + 2];
          }
        out[i * 3] = red / sum;
        out[i * 3 + 1] = green / sum;
        out[i * 3 + 2] = blue / sum;
      }
    return out;
  }

  function parametricKernel(model, amount, angle = 0) {
    const size = odd(model === "motion" ? amount + 4 : amount * 2 + 4),
      k = new Float32Array(size * size),
      center = (size - 1) / 2;
    if (model === "defocus" && amount < 0.5) {
      k[center * size + center] = 1;
    } else if (model === "motion") {
      const theta = (angle * Math.PI) / 180,
        samples = Math.max(16, Math.ceil(amount * 8));
      for (let i = 0; i < samples; i++) {
        const t = ((i + 0.5) / samples - 0.5) * amount;
        const x = center + Math.cos(theta) * t,
          y = center + Math.sin(theta) * t,
          ix = Math.floor(x),
          iy = Math.floor(y);
        for (let dy = 0; dy < 2; dy++)
          for (let dx = 0; dx < 2; dx++) {
            k[(iy + dy) * size + ix + dx] +=
              (dx ? x - ix : 1 - x + ix) * (dy ? y - iy : 1 - y + iy);
          }
      }
    } else if (model === "defocus") {
      for (let y = 0; y < size; y++)
        for (let x = 0; x < size; x++) {
          // Subpixel coverage keeps small disks useful and normalized.
          for (let sy = 0; sy < 4; sy++)
            for (let sx = 0; sx < 4; sx++) {
              if (
                Math.hypot(
                  x - center + (sx + 0.5) / 4 - 0.5,
                  y - center + (sy + 0.5) / 4 - 0.5,
                ) <= amount
              )
                k[y * size + x] += 1 / 16;
            }
        }
    } else throw new Error("Unknown blur model.");
    return {
      k: normalizeKernel(k),
      size,
      mode: model,
      gamma: 1,
      candidateCount: 1,
    };
  }
  function rgbaData(rgba, w, h, maxSide = Infinity) {
    const scale = Math.min(1, maxSide / Math.max(w, h)),
      nw = Math.max(1, Math.round(w * scale)),
      nh = Math.max(1, Math.round(h * scale));
    const rgb = new Float32Array(nw * nh * 3);
    // Area averaging avoids aliasing and uses every source pixel in the estimate.
    for (let y = 0; y < nh; y++)
      for (let x = 0; x < nw; x++) {
        const x0 = (x * w) / nw,
          x1 = ((x + 1) * w) / nw,
          y0 = (y * h) / nh,
          y1 = ((y + 1) * h) / nh;
        let red = 0,
          green = 0,
          blue = 0,
          total = 0;
        for (let sy = Math.floor(y0); sy < Math.ceil(y1); sy++)
          for (let sx = Math.floor(x0); sx < Math.ceil(x1); sx++) {
            const weight =
              (Math.min(x1, sx + 1) - Math.max(x0, sx)) *
              (Math.min(y1, sy + 1) - Math.max(y0, sy));
            const i = (Math.min(h - 1, sy) * w + Math.min(w - 1, sx)) * 4;
            red += rgba[i] * weight;
            green += rgba[i + 1] * weight;
            blue += rgba[i + 2] * weight;
            total += weight;
          }
        const j = (y * nw + x) * 3;
        rgb[j] = red / (255 * total);
        rgb[j + 1] = green / (255 * total);
        rgb[j + 2] = blue / (255 * total);
      }
    return { rgb, gray: rgbToGray(rgb), w: nw, h: nh, scale };
  }
  function resizeRGBA(rgba, w, h, maxSide) {
    if (Math.max(w, h) <= maxSide) return { rgba, w, h, scale: 1 };
    const data = rgbaData(rgba, w, h, maxSide),
      out = new Uint8ClampedArray(data.w * data.h * 4);
    for (let y = 0; y < data.h; y++)
      for (let x = 0; x < data.w; x++) {
        const i = y * data.w + x,
          src =
            (Math.min(h - 1, Math.floor(((y + 0.5) * h) / data.h)) * w +
              Math.min(w - 1, Math.floor(((x + 0.5) * w) / data.w))) *
            4;
        for (let c = 0; c < 3; c++)
          out[i * 4 + c] = Math.round(255 * data.rgb[i * 3 + c]);
        out[i * 4 + 3] = rgba[src + 3];
      }
    return { rgba: out, w: data.w, h: data.h, scale: data.scale };
  }
  async function estimateKernel(source, options, progress) {
    if (options.model !== "auto")
      return parametricKernel(
        options.model,
        options.model === "motion" ? options.length : options.radius,
        options.angle,
      );
    const base = rgbaData(
      source.rgba,
      source.w,
      source.h,
      options.quality === "fast" ? 192 : 320,
    );
    if (
      Math.min(base.w, base.h) < 24 ||
      highpass(base.gray, base.w, base.h) < 0.0005
    ) {
      const k = new Float32Array(9);
      k[4] = 1;
      return { k, size: 3, mode: "identity", gamma: 1, candidateCount: 1 };
    }
    const scene = analyzeScene(base),
      limit = Math.min(125, Math.floor(Math.min(source.w, source.h) * 0.3));
    const supports = [9, 17, 29, 45, 65, 95, 125].filter((n) => n <= limit);
    if (!supports.length) supports.push(odd(Math.max(3, limit - 1)));
    const candidates = [],
      modes =
        options.quality === "fast"
          ? [scene.lowLight || scene.highSaturation ? "gradient" : "dark"]
          : ["gradient", "dark"];
    let done = 0;
    for (const support of supports)
      for (const mode of modes) {
        progress(
          `Estimating blur: ${support} px, ${mode === "dark" ? "dark-channel" : "gradient"} search`,
          5 + (30 * done) / (supports.length * modes.length),
        );
        await nextFrame();
        candidates.push(await blindCandidate(base, support, mode, 1, 2));
        done++;
      }
    candidates.sort((a, b) => a.score.value - b.score.value);
    // Validate refined kernels on the same observations. A longer run is not
    // automatically better; retain the coarse solution when refinement regresses.
    const finalists = [
      ...candidates.slice(0, options.quality === "fast" ? 1 : 2),
    ];
    for (const candidate of finalists) {
      progress(`Refining ${candidate.fullSupport} px blur estimate`, 38);
      const refined = await blindCandidate(
        base,
        candidate.fullSupport,
        candidate.mode,
        candidate.gamma,
        options.quality === "fast" ? 3 : 5,
      );
      candidates.push(refined);
    }
    candidates.sort((a, b) => a.score.value - b.score.value);
    const winner = candidates[0];
    return {
      ...winner,
      k: resizeKernel(winner.k, winner.size, winner.fullSupport),
      size: winner.fullSupport,
      candidateCount: candidates.length,
    };
  }
  async function restoreTile(data, k, ks, regs, progress) {
    let t = performance.now();
    const base = chooseBaseline(data, k, ks, null, regs.baseline);
    const baseline = { ...base, kernel: k, ks, runtime: performance.now() - t };
    progress("Restoring motion candidate");
    await nextFrame();
    t = performance.now();
    const mk =
      kernelStats(k, ks).anis >= 0.55 ? motionConstrainKernel(k, ks) : k;
    const motion = {
      ...chooseBaseline(data, mk, ks, null, regs.motion),
      kernel: mk,
      ks,
      runtime: performance.now() - t,
    };
    progress("Refining with an edge-preserving prior");
    await nextFrame();
    t = performance.now();
    const pnp = {
      ...safeBlend(
        data.rgb,
        pnpRefine(data.rgb, base.img, data.w, data.h, k, ks, base.reg),
        data.w,
        data.h,
        k,
        ks,
        base.img,
      ),
      kernel: k,
      ks,
      runtime: performance.now() - t,
    };
    progress("Refining dark and bright details");
    await nextFrame();
    t = performance.now();
    const ext = {
      ...safeBlend(
        data.rgb,
        extremaRefine(data.rgb, base.img, data.w, data.h, k, ks, base.reg),
        data.w,
        data.h,
        k,
        ks,
        base.img,
      ),
      kernel: k,
      ks,
      runtime: performance.now() - t,
    };
    progress("Combining restoration candidates");
    await nextFrame();
    t = performance.now();
    // Compare every candidate using the same forward model, even when the motion
    // candidate was restored with a different PSF.
    const candidates = [base, motion, pnp, ext].map((item) => ({
      ...item,
      q: quality(data.rgb, item.img, data.w, data.h, k, ks),
    }));
    candidates.sort((a, b) => a.q.score - b.q.score);
    const fusion = rgacRefine(data.rgb, candidates, data.w, data.h, k, ks);
    const rgac = {
      ...safeBlend(
        data.rgb,
        fusion.img,
        data.w,
        data.h,
        k,
        ks,
        candidates[0].img,
      ),
      kernel: k,
      ks,
      runtime: performance.now() - t,
    };
    return {
      baseline,
      motion_constrained: motion,
      annealed_pnp: pnp,
      extreme_channel: ext,
      rgac,
    };
  }
  function tileStarts(length, size, overlap) {
    const positions = [0];
    while (positions[positions.length - 1] + size < length)
      positions.push(positions[positions.length - 1] + size - overlap);
    return positions;
  }
  function tileWeight(position, length, start, total, overlap) {
    let weight = 1;
    if (start > 0 && position < overlap)
      weight *= 0.5 - 0.5 * Math.cos((Math.PI * (position + 0.5)) / overlap);
    if (start + length < total && position >= length - overlap)
      weight *=
        0.5 - 0.5 * Math.cos((Math.PI * (length - position - 0.5)) / overlap);
    return Math.max(1e-6, weight);
  }
  async function restoreImage(source, estimate, options, progress) {
    const { w, h, rgba } = source,
      n = w * h,
      ids = [
        "baseline",
        "motion_constrained",
        "annealed_pnp",
        "extreme_channel",
        "rgac",
      ];
    const ks = estimate.size,
      k = estimate.k;
    // Tune on a representative sample, then use one regularization across tiles.
    // Estimate sensor noise at source resolution, never on a smoothed thumbnail.
    const cw = Math.min(w, 256),
      ch = Math.min(h, 256),
      crop = new Float32Array(cw * ch);
    const ox = (w - cw) >> 1,
      oy = (h - ch) >> 1;
    for (let y = 0; y < ch; y++)
      for (let x = 0; x < cw; x++) {
        const i = ((y + oy) * w + x + ox) * 4;
        crop[y * cw + x] =
          (0.2989360213 * rgba[i] +
            0.5870430745 * rgba[i + 1] +
            0.1140209043 * rgba[i + 2]) /
          255;
      }
    const noise = noiseEstimate(crop, cw, ch),
      reg = clamp(0.0006 + 8 * noise * noise, 0.0006, 0.04) * options.denoise;
    const regs = { baseline: reg, motion: reg },
      weights = new Float32Array(n),
      methods = {};
    for (const id of ids)
      methods[id] = {
        rgba: new Uint8ClampedArray(n * 4),
        kernel:
          id === "motion_constrained" && kernelStats(k, ks).anis >= 0.55
            ? motionConstrainKernel(k, ks)
            : k,
        ks,
        runtime: 0,
      };
    const size = options.tileSize || 384,
      overlap = Math.min(96, Math.max(32, ks));
    const xs = tileStarts(w, size, overlap),
      ys = tileStarts(h, size, overlap),
      total = xs.length * ys.length;
    // A halo outside each blended tile isolates FFT and prior boundary effects.
    const halo = Math.max(16, ks),
      started = performance.now();
    let completed = 0;
    for (const y0 of ys)
      for (const x0 of xs) {
        const tw = Math.min(size, w - x0),
          th = Math.min(size, h - y0);
        // At a physical image edge let the solver apply the reflected model.
        // Reflecting observations into a halo first changes angled blur there.
        const left = Math.max(0, x0 - halo),
          top = Math.max(0, y0 - halo);
        const ew = Math.min(w, x0 + tw + halo) - left,
          eh = Math.min(h, y0 + th + halo) - top,
          rgb = new Float32Array(ew * eh * 3);
        for (let y = 0; y < eh; y++)
          for (let x = 0; x < ew; x++) {
            const src = ((top + y) * w + left + x) * 4,
              i = (y * ew + x) * 3;
            for (let c = 0; c < 3; c++) rgb[i + c] = rgba[src + c] / 255;
          }
        const data = { rgb, gray: rgbToGray(rgb), w: ew, h: eh };
        const tiles = await restoreTile(data, k, ks, regs, (text) =>
          progress(
            `${text} · tile ${completed + 1}/${total}`,
            45 + (49 * completed) / total,
          ),
        );
        for (let y = 0; y < th; y++)
          for (let x = 0; x < tw; x++) {
            const index = (y0 + y) * w + x0 + x,
              ti = ((y + y0 - top) * ew + x + x0 - left) * 3;
            const weight =
                tileWeight(x, tw, x0, w, overlap) *
                tileWeight(y, th, y0, h, overlap),
              sum = weights[index] + weight;
            for (const id of ids) {
              const output = methods[id].rgba,
                tile = tiles[id].img;
              for (let c = 0; c < 3; c++)
                output[index * 4 + c] = Math.round(
                  (output[index * 4 + c] * weights[index] +
                    255 * clamp(tile[ti + c]) * weight) /
                    sum,
                );
              output[index * 4 + 3] = rgba[index * 4 + 3];
            }
            weights[index] = sum;
          }
        for (const id of ids) methods[id].runtime += tiles[id].runtime;
        completed++;
        await nextFrame();
      }
    progress("Measuring the completed results", 96);
    const diagnostic = rgbaData(rgba, w, h, 320),
      metricSize = odd(Math.max(3, Math.round(ks * diagnostic.scale)));
    // All recommendation scores use the same PSF and same diagnostic resolution.
    const metricKernel = resizeKernel(k, ks, metricSize);
    for (const id of ids) {
      const candidate = rgbaData(methods[id].rgba, w, h, 320);
      methods[id].q = quality(
        diagnostic.rgb,
        candidate.rgb,
        candidate.w,
        candidate.h,
        metricKernel,
        metricSize,
      );
    }
    const recommended = ids.reduce((a, b) =>
      methods[a].q.score <= methods[b].q.score ? a : b,
    );
    return {
      methods,
      w,
      h,
      recommended,
      noise,
      reg,
      tiles: total,
      diagnosticSize: `${diagnostic.w}×${diagnostic.h}`,
      restoreRuntime: performance.now() - started,
    };
  }
  function validateOptions(input) {
    const options = {
      quality: "quality",
      resolution: "native",
      model: "auto",
      length: 17,
      angle: 0,
      radius: 3,
      denoise: 1,
      ...input,
    };
    if (
      !["quality", "fast"].includes(options.quality) ||
      !["native", "preview"].includes(options.resolution) ||
      !["auto", "motion", "defocus"].includes(options.model)
    )
      throw new Error("Invalid processing options.");
    for (const [key, low, high] of [
      ["length", 1, 121],
      ["angle", -180, 180],
      ["radius", 0.5, 30],
      ["denoise", 0.25, 4],
    ]) {
      if (
        !Number.isFinite(options[key]) ||
        options[key] < low ||
        options[key] > high
      )
        throw new Error(`Invalid ${key}: expected ${low}–${high}.`);
    }
    if (
      options.tileSize !== undefined &&
      (!Number.isInteger(options.tileSize) ||
        options.tileSize < 128 ||
        options.tileSize > 512)
    )
      throw new Error("Invalid tile size.");
    return options;
  }
  async function run(input, suppliedOptions = {}, onProgress = () => {}) {
    const options = validateOptions(suppliedOptions),
      started = performance.now();
    const { w, h, rgba } = input;
    if (
      !Number.isInteger(w) ||
      !Number.isInteger(h) ||
      w < 1 ||
      h < 1 ||
      w > 16384 ||
      h > 16384 ||
      w * h > 40000000 ||
      !rgba ||
      rgba.length !== w * h * 4
    )
      throw new Error(
        "Invalid image dimensions or pixel buffer. Use an image up to 40 megapixels and 16,384 pixels per side.",
      );
    if (options.resolution === "native" && w * h > 12000000)
      throw new Error(
        "Native processing supports up to 12 megapixels. Choose 1400 px preview or a smaller image; the original is never silently reduced.",
      );
    onProgress("Preparing image analysis", 2);
    await nextFrame();
    const source =
      options.resolution === "preview"
        ? resizeRGBA(rgba, w, h, 1400)
        : { rgba, w, h, scale: 1 };
    // Manual controls are expressed in original-image pixels, even in preview.
    const scaledOptions = {
      ...options,
      length: options.length * source.scale,
      radius: options.radius * source.scale,
    };
    const estimate = await estimateKernel(source, scaledOptions, onProgress);
    onProgress("Blur estimated; preparing restoration", 43);
    await nextFrame();
    const result = await restoreImage(source, estimate, options, onProgress);
    return {
      ...result,
      estimate,
      options,
      sourceWidth: w,
      sourceHeight: h,
      scale: source.scale,
      totalRuntime: performance.now() - started,
    };
  }
  function adjointBlur(obs, w, h, k, ks) {
    // A = crop * convolution * reflect. Its adjoint folds boundary contributions
    // back to their source pixels; simply flipping a reflected kernel is wrong.
    const layout = fftLayout(w, h, ks),
      { pw, ph, padX, padY } = layout;
    const re = new Float64Array(pw * ph),
      im = new Float64Array(pw * ph);
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++)
        re[(y + padY) * pw + x + padX] = obs[y * w + x];
    fft2(re, im, pw, ph, false);
    const H = transferFunction(k, ks, pw, ph);
    for (let i = 0; i < re.length; i++) {
      const r = re[i],
        q = im[i];
      re[i] = r * H.re[i] + q * H.im[i];
      im[i] = q * H.re[i] - r * H.im[i];
    }
    fft2(re, im, pw, ph, true);
    const out = new Float32Array(w * h),
      radius = (ks - 1) >> 1;
    for (let y = -radius; y < h + radius; y++)
      for (let x = -radius; x < w + radius; x++)
        out[reflect(y, h) * w + reflect(x, w)] +=
          re[(y + padY) * pw + x + padX];
    return out;
  }
  function deconvChannel(obs, w, h, k, ks, reg, prior = null, rho = 0) {
    const x = fftDeconvChannel(obs, w, h, k, ks, reg, prior, rho);
    const apply = (values) => {
      const out = adjointBlur(blurChannel(values, w, h, k, ks), w, h, k, ks);
      for (let y = 0; y < h; y++)
        for (let col = 0; col < w; col++) {
          const i = y * w + col,
            v = values[i];
          let lap = 0;
          if (col > 0) lap += v - values[i - 1];
          if (col + 1 < w) lap += v - values[i + 1];
          if (y > 0) lap += v - values[i - w];
          if (y + 1 < h) lap += v - values[i + w];
          out[i] += reg * lap + (prior ? rho * v : 0);
        }
      return out;
    };
    const rhs = adjointBlur(obs, w, h, k, ks),
      Ax = apply(x),
      r = new Float32Array(x.length);
    let rs = 0;
    for (let i = 0; i < x.length; i++) {
      r[i] = rhs[i] + (prior ? rho * prior[i] : 0) - Ax[i];
      rs += r[i] * r[i];
    }
    const precondition = (residual) => {
      const layout = fftLayout(w, h, ks),
        { pw, ph, padX, padY } = layout;
      const re = new Float64Array(pw * ph),
        im = new Float64Array(pw * ph);
      for (let y = 0; y < h; y++)
        for (let col = 0; col < w; col++)
          re[(y + padY) * pw + col + padX] = residual[y * w + col];
      fft2(re, im, pw, ph, false);
      const H = transferFunction(k, ks, pw, ph);
      for (let i = 0; i < re.length; i++) {
        const den = Math.max(
          1e-8,
          H.re[i] ** 2 + H.im[i] ** 2 + reg * H.penalty[i] + (prior ? rho : 0),
        );
        re[i] /= den;
        im[i] /= den;
      }
      fft2(re, im, pw, ph, true);
      return cropTransform(re, w, h, layout);
    };
    let z = precondition(r),
      rz = 0;
    for (let i = 0; i < r.length; i++) rz += r[i] * z[i];
    const p = new Float32Array(z),
      tolerance = Math.max(1e-14, rs * 1e-6);
    // Spectrally preconditioned CG corrects the warm start against the actual
    // reflected forward model, including angled/asymmetric blur at boundaries.
    for (let step = 0; step < (prior ? 3 : 16) && rs > tolerance; step++) {
      const Ap = apply(p);
      let denom = 0;
      for (let i = 0; i < p.length; i++) denom += p[i] * Ap[i];
      if (denom <= 1e-20) break;
      const alpha = rz / denom;
      rs = 0;
      for (let i = 0; i < x.length; i++) {
        x[i] += alpha * p[i];
        r[i] -= alpha * Ap[i];
        rs += r[i] * r[i];
      }
      if (rs <= tolerance) break;
      z = precondition(r);
      let next = 0;
      for (let i = 0; i < r.length; i++) next += r[i] * z[i];
      const beta = next / Math.max(rz, 1e-30);
      for (let i = 0; i < p.length; i++) p[i] = z[i] + beta * p[i];
      rz = next;
    }
    return x;
  }

  root.DeblurCore = {
    adjointBlur,
    run,
    validateOptions,
    parametricKernel,
    blurChannel,
    deconvChannel,
    restoreRGB,
    quality,
    safeBlend,
    noiseEstimate,
    localMinProjection,
    rgbaData,
    resizeRGBA,
    resizeKernel,
    kernelStats,
    tileStarts,
    tileWeight,
    fft1,
    fft2,
    normalizeKernel,
    chooseBaseline,
    pnpRefine,
    rgacRefine,
    estimateKernel,
  };
  if (typeof module !== "undefined" && module.exports)
    module.exports = root.DeblurCore;
})(typeof self !== "undefined" ? self : globalThis);
