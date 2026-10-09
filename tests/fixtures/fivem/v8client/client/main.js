addEventListener("v8:ready", () => {});
onNet("v8:net", function () {});
RegisterCommand("v8cmd", () => {
  emit("v8:ready");
  emitNet("v8:net");
  TriggerEvent("v8:ready");
  TriggerLatentServerEvent("v8:bulk", 4096);
}, false);
exports("v8export", () => 1);
