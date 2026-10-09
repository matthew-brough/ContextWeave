using System;
using CitizenFX.Core;

namespace Duel.Client
{
    public class ClientScript : BaseScript
    {
        public ClientScript()
        {
            EventHandlers["duel:csClient"] += new Action<string>((msg) =>
            {
                TriggerServerEvent("duel:serverHello", msg);
                TriggerServerEvent("duel:both");
            });
            EventHandlers["duel:csBoth"] += new Action(() => { });
        }
    }
}
