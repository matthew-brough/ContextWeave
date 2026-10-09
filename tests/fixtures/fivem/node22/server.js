import { helper } from "./lib/index.js";

onNet("n22:ping", async () => {
  helper("pong");
  emitNet("n22:pong", source);
  emitNet("duel:csClient", source);
});

on("n22:tick", () => {});
TriggerEvent("duel:sharedHook");
