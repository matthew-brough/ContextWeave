import { wire } from "@n26/netkit";
import { extra } from "@n26/netkit/extra";
export { registerAll } from "./lib/handlers.js";

onNet("n26:ping", () => {
  wire("pong");
  emitNet("n26:pong", source, extra);
  emitNet("n26:late", source);
});

const lazy = () => import("./lib/handlers.js");
