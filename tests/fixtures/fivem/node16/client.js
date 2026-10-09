onNet("n16:pong", (n) => {
  console.log(n);
});

AddEventHandler("n16:local", () => {});

socket.on("n16:pong", () => {});
emitNet("n16:ping");
TriggerServerEvent("n16:ping");
