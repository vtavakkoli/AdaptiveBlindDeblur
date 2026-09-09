"use strict";
importScripts("deblur-core.js?v=large-kernel-1");
self.onmessage = async ({ data }) => {
  const { id, image, options } = data;
  try {
    const result = await self.DeblurCore.run(image, options, (text, percent) =>
      self.postMessage({ id, type: "progress", text, percent }),
    );
    const transfers = Object.values(result.methods).map(
      (item) => item.rgba.buffer,
    );
    self.postMessage({ id, type: "result", result }, transfers);
  } catch (error) {
    self.postMessage({
      id,
      type: "error",
      message: error.message || String(error),
    });
  }
};
