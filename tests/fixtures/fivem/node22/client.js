import { helper } from "./lib/util.js";

RegisterNetEvent("n22:pong", () => {
  helper("x");
});

emitNet("n22:ping");
emit("n22:tick");
