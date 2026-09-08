"use strict";
const METHOD_META = {
  baseline: {
    name: "Adaptive Robust Baseline",
    description:
      "Noise-aware deconvolution with a guard against ringing and new clipping.",
  },
  motion_constrained: {
    name: "Motion-Constrained",
    description:
      "Constrains directional blur to a motion corridor. Isotropic blur keeps its original kernel.",
  },
  annealed_pnp: {
    name: "Annealed PnP",
    description:
      "An edge-preserving denoising prior alternates with blur-consistent restoration.",
  },
  extreme_channel: {
    name: "Dual-Extreme",
    description: "Refines detail around informative dark and bright regions.",
  },
  rgac: {
    name: "RGAC",
    description:
      "Combines complementary candidates evaluated against a shared blur model.",
  },
};
function init() {
  const ids = [
    "fileInput",
    "dropZone",
    "runBtn",
    "cancelBtn",
    "statusText",
    "progressBar",
    "imageMetric",
    "outputMetric",
    "runtimeMetric",
    "kernelMetric",
    "methodGrid",
    "recommendation",
    "methodTitle",
    "methodDescription",
    "viewer",
    "viewerScroll",
    "emptyViewer",
    "originalImage",
    "resultImage",
    "beforeLabel",
    "afterLabel",
    "splitLine",
    "beforeAfterSlider",
    "exportBtn",
    "reportBtn",
    "rmseMetric",
    "edgeMetric",
    "highpassMetric",
    "clipMetric",
    "methodScoreMetric",
    "methodRuntimeMetric",
    "psfCanvas",
    "psfNote",
    "kernelDecision",
    "metricNote",
    "qualitySelect",
    "resolutionSelect",
    "modelSelect",
    "motionControls",
    "defocusControls",
    "motionLength",
    "motionAngle",
    "defocusRadius",
    "denoise",
    "denoiseValue",
    "zoomSelect",
    "settings",
    "fileName",
  ];
  const E = Object.fromEntries(
    ids.map((id) => [id, document.getElementById(id)]),
  );
  if (Object.values(E).some((value) => !value) || !globalThis.DeblurCore) {
    document.getElementById("fatal").textContent =
      "The app could not load. Keep all Browser Lab files together and reload the page.";
    document.getElementById("fatal").hidden = false;
    return;
  }
  const S = {
    image: null,
    sourceUrl: null,
    resultUrl: null,
    fileName: "image",
    selected: "rgac",
    result: null,
    worker: null,
    generation: 0,
    renderId: 0,
    busy: false,
    decoding: false,
    cancelJob: null,
  };
  const clamp = (v, a = 0, b = 1) => Math.max(a, Math.min(b, v));
  function status(text, percent = 0) {
    E.statusText.textContent = text;
    E.progressBar.value = clamp(percent, 0, 100);
  }
  function busy(value) {
    S.busy = value;
    E.runBtn.disabled = value || S.decoding || !S.image;
    E.runBtn.textContent = value ? "Restoring…" : "Analyze & deblur";
    E.cancelBtn.hidden = !value;
    E.settings.disabled = value;
    E.viewerScroll.setAttribute("aria-busy", String(value));
  }
  function revoke(key) {
    if (S[key]) URL.revokeObjectURL(S[key]);
    S[key] = null;
  }
  function reset() {
    S.renderId++;
    S.result = null;
    revoke("resultUrl");
    E.resultImage.hidden = true;
    E.resultImage.removeAttribute("src");
    for (const id of [
      "beforeLabel",
      "afterLabel",
      "splitLine",
      "beforeAfterSlider",
    ])
      E[id].hidden = true;
    for (const id of [
      "outputMetric",
      "runtimeMetric",
      "kernelMetric",
      "rmseMetric",
      "edgeMetric",
      "highpassMetric",
      "clipMetric",
      "methodScoreMetric",
      "methodRuntimeMetric",
    ])
      E[id].textContent = "—";
    E.recommendation.hidden = true;
    E.exportBtn.disabled = true;
    E.reportBtn.disabled = true;
    E.metricNote.textContent =
      "Diagnostics describe the restored pixels. A lower score does not prove better visual quality.";
    E.psfNote.textContent = "The blur kernel will appear after processing.";
    E.kernelDecision.textContent =
      "Choose automatic estimation, or specify motion or defocus blur.";
    document
      .querySelectorAll(".method-option")
      .forEach((el) => el.classList.remove("recommended"));
    drawKernel(null, 0);
  }
  function cancel(announce = true) {
    S.generation++;
    S.renderId++;
    S.worker?.terminate();
    S.worker = null;
    if (S.cancelJob) {
      S.cancelJob(new Error("Cancelled"));
      S.cancelJob = null;
    }
    busy(false);
    if (announce)
      status("Cancelled. You can change the settings and run again.");
  }
  async function loadFile(file) {
    if (!file) return;
    if (
      !/^image\/(png|jpeg|webp)$/.test(file.type) &&
      !/\.(png|jpe?g|webp)$/i.test(file.name)
    ) {
      status("Choose a PNG, JPEG or WebP image.");
      return;
    }
    if (file.size > 50 * 1024 * 1024) {
      status("Choose an image file smaller than 50 MB.");
      return;
    }
    cancel(false);
    const generation = S.generation,
      url = URL.createObjectURL(file),
      img = new Image();
    S.decoding = true;
    busy(false);
    status("Opening image…");
    try {
      img.src = url;
      await img.decode();
      if (generation !== S.generation) {
        URL.revokeObjectURL(url);
        return;
      }
      if (
        img.naturalWidth > 16384 ||
        img.naturalHeight > 16384 ||
        img.naturalWidth * img.naturalHeight > 40000000
      )
        throw new Error(
          "Choose an image up to 40 megapixels and 16,384 pixels per side.",
        );
      revoke("sourceUrl");
      S.sourceUrl = url;
      S.image = img;
      S.fileName = file.name.replace(/\.[^.]+$/, "");
      reset();
      E.originalImage.src = url;
      E.originalImage.hidden = false;
      E.emptyViewer.hidden = true;
      E.fileName.textContent = file.name;
      E.imageMetric.textContent = `${img.naturalWidth} × ${img.naturalHeight}`;
      E.zoomSelect.value = "fit";
      resizeViewer();
      status(
        img.naturalWidth * img.naturalHeight > 12000000
          ? "Image ready. For images above 12 MP, select the 1400 px preview."
          : "Image ready. Native output preserves its dimensions. Larger images take longer.",
      );
    } catch (error) {
      URL.revokeObjectURL(url);
      if (generation === S.generation)
        status(error.message || "This image could not be decoded.");
    } finally {
      if (generation === S.generation) {
        S.decoding = false;
        busy(false);
      }
    }
  }
  E.fileInput.addEventListener("change", () => {
    loadFile(E.fileInput.files?.[0]);
    E.fileInput.value = "";
  });
  for (const name of ["dragenter", "dragover"])
    E.dropZone.addEventListener(name, (event) => {
      event.preventDefault();
      E.dropZone.classList.add("drag");
    });
  for (const name of ["dragleave", "drop"])
    E.dropZone.addEventListener(name, (event) => {
      event.preventDefault();
      E.dropZone.classList.remove("drag");
    });
  E.dropZone.addEventListener("drop", (event) =>
    loadFile(event.dataTransfer?.files?.[0]),
  );
  E.cancelBtn.addEventListener("click", () => cancel());
  function updateModel() {
    E.motionControls.hidden = E.modelSelect.value !== "motion";
    E.defocusControls.hidden = E.modelSelect.value !== "defocus";
  }
  E.modelSelect.addEventListener("change", updateModel);
  E.denoise.addEventListener("input", () => {
    E.denoiseValue.value = `${Number(E.denoise.value).toFixed(2)}×`;
  });
  E.settings.addEventListener("change", () => {
    if (S.result)
      status(
        "Settings changed. Run again to apply them; the current result still uses the previous settings.",
        100,
      );
  });
  function resizeViewer() {
    if (!S.image) return;
    const w = S.result?.w || S.image.naturalWidth,
      h = S.result?.h || S.image.naturalHeight;
    const available = Math.max(1, E.viewerScroll.clientWidth - 2),
      maxHeight = Math.max(260, Math.min(720, window.innerHeight * 0.65));
    const width =
      E.zoomSelect.value === "fit"
        ? Math.min(available, (maxHeight * w) / h)
        : w * Number(E.zoomSelect.value);
    E.viewer.style.width = `${width}px`;
    E.viewer.style.height = `${(width * h) / w}px`;
    E.viewerScroll.classList.toggle("zoomed", E.zoomSelect.value !== "fit");
  }
  E.zoomSelect.addEventListener("change", resizeViewer);
  window.addEventListener("resize", resizeViewer);
  function updateSplit() {
    const value = clamp(Number(E.beforeAfterSlider.value), 0, 100);
    E.resultImage.style.clipPath = `inset(0 0 0 ${value}%)`;
    E.splitLine.style.left = `${value}%`;
    E.beforeAfterSlider.setAttribute(
      "aria-valuetext",
      `${value}% original, ${100 - value}% restored`,
    );
  }
  E.beforeAfterSlider.addEventListener("input", updateSplit);
  let dragging = false;
  function moveSplit(event) {
    const rect = E.viewer.getBoundingClientRect();
    E.beforeAfterSlider.value = String(
      Math.round(
        100 * clamp((event.clientX - rect.left) / Math.max(1, rect.width)),
      ),
    );
    updateSplit();
  }
  E.viewer.addEventListener("pointerdown", (event) => {
    if (
      !S.result ||
      (event.pointerType === "touch" && E.zoomSelect.value !== "fit")
    )
      return;
    dragging = true;
    E.viewer.setPointerCapture?.(event.pointerId);
    moveSplit(event);
  });
  E.viewer.addEventListener("pointermove", (event) => {
    if (dragging) moveSplit(event);
  });
  for (const name of ["pointerup", "pointercancel", "lostpointercapture"])
    E.viewer.addEventListener(name, () => {
      dragging = false;
    });
  function drawKernel(k, size) {
    const canvas = E.psfCanvas,
      ctx = canvas.getContext("2d");
    ctx.fillStyle = "#101828";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    if (!k) return;
    const pixels = ctx.createImageData(canvas.width, canvas.height);
    let peak = 0;
    for (const v of k) peak = Math.max(peak, v);
    for (let y = 0; y < canvas.height; y++)
      for (let x = 0; x < canvas.width; x++) {
        const index = (y * canvas.width + x) * 4;
        const value = Math.pow(
          k[
            Math.floor((y * size) / canvas.height) * size +
              Math.floor((x * size) / canvas.width)
          ] / Math.max(1e-12, peak),
          0.5,
        );
        pixels.data[index] = 16 + 210 * value;
        pixels.data[index + 1] = 24 + 210 * value;
        pixels.data[index + 2] = 40 + 215 * value;
        pixels.data[index + 3] = 255;
      }
    ctx.putImageData(pixels, 0, 0);
  }
  async function renderMethod(id) {
    S.selected = id;
    E.methodTitle.textContent = METHOD_META[id].name;
    E.methodDescription.textContent = METHOD_META[id].description;
    const item = S.result?.methods[id];
    if (!item) return;
    const renderId = ++S.renderId,
      generation = S.generation,
      canvas = document.createElement("canvas");
    E.exportBtn.disabled = true;
    try {
      canvas.width = S.result.w;
      canvas.height = S.result.h;
      canvas
        .getContext("2d")
        .putImageData(
          new ImageData(item.rgba, canvas.width, canvas.height),
          0,
          0,
        );
      const blob = await new Promise((resolve) =>
        canvas.toBlob(resolve, "image/png"),
      );
      canvas.width = canvas.height = 1;
      if (!blob)
        throw new Error(
          "PNG export could not be created. Try preview resolution.",
        );
      if (renderId !== S.renderId || generation !== S.generation) return;
      revoke("resultUrl");
      S.resultUrl = URL.createObjectURL(blob);
      E.resultImage.src = S.resultUrl;
      E.resultImage.hidden = false;
      for (const key of [
        "beforeLabel",
        "afterLabel",
        "splitLine",
        "beforeAfterSlider",
      ])
        E[key].hidden = false;
      E.exportBtn.disabled = false;
      E.reportBtn.disabled = false;
      updateSplit();
      resizeViewer();
      E.rmseMetric.textContent = item.q.rm.toFixed(5);
      E.edgeMetric.textContent = `${item.q.edge.toFixed(2)}×`;
      E.highpassMetric.textContent = `${item.q.hp.toFixed(2)}×`;
      E.clipMetric.textContent = `${(100 * item.q.extraClip).toFixed(1)}%`;
      E.methodScoreMetric.textContent = item.q.score.toFixed(5);
      E.methodRuntimeMetric.textContent = `${(item.runtime / 1000).toFixed(1)} s`;
      drawKernel(item.kernel, item.ks);
      E.psfNote.textContent = `${item.ks} × ${item.ks} kernel in output pixels. ${S.result.options.model === "auto" ? "Estimated from this image." : "Using your blur settings."}`;
    } catch (error) {
      if (renderId === S.renderId) status(error.message);
    }
  }
  E.methodGrid
    .querySelectorAll("input")
    .forEach((input) =>
      input.addEventListener("change", () => renderMethod(input.value)),
    );
  function download(url, name) {
    const a = document.createElement("a");
    a.href = url;
    a.download = name;
    a.click();
  }
  E.exportBtn.addEventListener("click", () => {
    if (S.resultUrl)
      download(
        S.resultUrl,
        `${S.fileName}-${S.selected}-${S.result.w}x${S.result.h}.png`,
      );
  });
  E.reportBtn.addEventListener("click", () => {
    if (!S.result) return;
    const { methods, estimate, ...metadata } = S.result;
    const report = {
      schema: "browser-lab-v2",
      ...metadata,
      selectedMethod: S.selected,
      note: "Reference-free diagnostics at the stated diagnosticSize using the common estimated PSF. Not ground-truth quality or Python benchmark results.",
      estimate: { ...estimate, k: Array.from(estimate.k) },
      methods: Object.fromEntries(
        Object.entries(methods).map(([id, item]) => [
          id,
          {
            q: item.q,
            runtime: item.runtime,
            kernelSize: item.ks,
            kernel: Array.from(item.kernel),
          },
        ]),
      ),
    };
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(report, null, 2)], { type: "application/json" }),
    );
    download(url, `${S.fileName}-deblur-settings.json`);
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  async function execute(image, options, generation) {
    const progress = (text, percent) => {
      if (generation !== S.generation) throw new Error("Cancelled");
      status(text, percent);
    };
    // file:// browsers commonly disallow workers. Keep the offline workflow with
    // cooperative yields; hosted pages always use a dedicated worker.
    if (location.protocol === "file:" || typeof Worker === "undefined") {
      status("Processing locally. This browser uses the compatibility mode.");
      return DeblurCore.run(image, options, progress);
    }
    return new Promise((resolve, reject) => {
      const worker = new Worker("deblur-worker.js");
      S.worker = worker;
      S.cancelJob = reject;
      worker.onmessage = ({ data }) => {
        if (generation !== S.generation || data.id !== generation) return;
        if (data.type === "progress") progress(data.text, data.percent);
        else if (data.type === "result") resolve(data.result);
        else if (data.type === "error") reject(new Error(data.message));
      };
      worker.onerror = () =>
        reject(
          new Error(
            "The processing worker could not run. Reload with all Browser Lab files present, or open index.html locally for compatibility mode.",
          ),
        );
      worker.postMessage({ id: generation, image, options }, [
        image.rgba.buffer,
      ]);
    });
  }
  E.runBtn.addEventListener("click", async () => {
    if (!S.image || S.busy || S.decoding) return;
    const generation = ++S.generation;
    try {
      const options = DeblurCore.validateOptions({
        quality: E.qualitySelect.value,
        resolution: E.resolutionSelect.value,
        model: E.modelSelect.value,
        length: Number(E.motionLength.value),
        angle: Number(E.motionAngle.value),
        radius: Number(E.defocusRadius.value),
        denoise: Number(E.denoise.value),
      });
      const w = S.image.naturalWidth,
        h = S.image.naturalHeight;
      if (w * h > 12000000 && options.resolution === "native")
        throw new Error(
          "Native output supports up to 12 MP. Select 1400 px preview or choose a smaller image.",
        );
      reset();
      busy(true);
      status("Preparing image…", 1);
      await new Promise((resolve) => setTimeout(resolve, 0));
      if (generation !== S.generation) return;
      const canvas = document.createElement("canvas");
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext("2d", { willReadFrequently: true });
      ctx.drawImage(S.image, 0, 0);
      const rgba = ctx.getImageData(0, 0, w, h).data;
      canvas.width = canvas.height = 1;
      const result = await execute({ rgba, w, h }, options, generation);
      if (generation !== S.generation) return;
      S.result = result;
      E.outputMetric.textContent = `${result.w} × ${result.h}${result.scale < 1 ? " · preview" : " · native"}`;
      E.runtimeMetric.textContent = `${(result.totalRuntime / 1000).toFixed(1)} s`;
      E.kernelMetric.textContent = `${result.estimate.size} × ${result.estimate.size}`;
      E.recommendation.textContent = `Lowest diagnostic score: ${METHOD_META[result.recommended].name}`;
      E.recommendation.hidden = false;
      document
        .querySelectorAll(".method-option")
        .forEach((el) =>
          el.classList.toggle(
            "recommended",
            el.dataset.method === result.recommended,
          ),
        );
      E.kernelDecision.textContent = `${result.estimate.mode} · ${result.estimate.candidateCount} candidates · ${result.tiles} restoration tiles.`;
      E.metricNote.textContent = `Diagnostics measured at ${result.diagnosticSize}, using one common blur model. Use 100% zoom to judge detail and ringing; scores are not a quality guarantee.`;
      await renderMethod(S.selected);
      if (generation === S.generation)
        status(
          `Complete: ${result.w} × ${result.h} ${result.scale < 1 ? "preview" : "native"} pixels. Compare the five methods and export the selected result.`,
          100,
        );
    } catch (error) {
      if (generation === S.generation) status(error.message || String(error));
    } finally {
      if (generation === S.generation) {
        S.worker?.terminate();
        S.worker = null;
        S.cancelJob = null;
        busy(false);
      }
    }
  });
  window.addEventListener("pagehide", (event) => {
    cancel(false);
    if (!event.persisted) {
      revoke("sourceUrl");
      revoke("resultUrl");
    }
  });
  updateModel();
  reset();
  busy(false);
  renderMethod(S.selected);
}
document.addEventListener("DOMContentLoaded", init, { once: true });
