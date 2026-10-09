onNet("ts:pong", (id: number): void => {
  emit("ts:tick");
});

emitNet("ts:ping", { id: 1 });
emit("ts:ui");
