using System;
using CitizenFX.Core;
using static CitizenFX.Core.Native.API;

namespace Sample.Net;

public class NetMain : BaseScript
{
    public NetMain()
    {
        EventHandlers["net:bare"] += (string name) => { };
        EventHandlers["net:cast"] += (Action<string>)((name) => { });
        EventHandlers["net:wrapped"] += new Action<string>((name) => { });
        EventHandlers["net:named"] += new Action<string>(OnNamed);
        EventHandlers["net:method"] += OnNamed;
        Exports.Add("netExport", new Func<int>(() => 1));
        RegisterCommand("netCommand", new Action<int, List<object>, string>((s, a, r) => { }), false);
        Query("SELECT 2", null, new Action<object>((rows) => { }));
    }

    private void OnNamed(string name) { }
}
