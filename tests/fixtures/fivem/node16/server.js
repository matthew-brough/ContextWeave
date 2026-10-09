const { helper } = require("./lib/util");

onNet("n16:ping", () => {
  helper("pong");
  emitNet("n16:pong", source, 1);
  emitNet("duel:clientPing", source);
  emit("n16:local");
});

on("n16:local", function () {
  console.log("local");
});

emitter.on("n16:ping", () => {});
emit("n16:" + helper("x"));
