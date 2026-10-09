using System;
using CitizenFX.Core;

namespace Duel.Server
{
    public class ServerScript : BaseScript
    {
        public ServerScript()
        {
            EventHandlers["duel:csServer"] += new Action<Player>((player) =>
            {
                TriggerClientEvent(player, "duel:clientPing", 1);
                TriggerClientEvent(player, "duel:csClient", "cs");
            });
        }
    }
}
