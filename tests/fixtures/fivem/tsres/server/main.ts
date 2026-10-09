interface Payload {
  id: number;
}

onNet("ts:ping", (payload: Payload): void => {
  emitNet("ts:pong", source, payload.id);
});

on("ts:tick", (): void => {});
