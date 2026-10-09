onNet("n26:pong", () => {});
emitNet("n26:ping");
addNetEventListener("n26:ping", () => {});
