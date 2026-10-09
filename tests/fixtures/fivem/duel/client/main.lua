RegisterNetEvent("duel:clientPing", function(from)
    print("ping from", from)
end)

RegisterNetEvent("duel:both", function()
    print("client both")
end)

AddEventHandler("duel:local", function()
    print("local")
end)

RegisterCommand("duel", function()
    TriggerServerEvent("duel:serverHello", 1)
    TriggerServerEvent("duel:csServer")
    TriggerLatentServerEvent("duel:bulk", 10240, {})
    TriggerEvent("duel:local")
    TriggerEvent("duel:both")
    TriggerServerEvent("duel:both")
    TriggerEvent("duel:sharedHook")
    TriggerServerEvent("duel:" .. GetCurrentResourceName() .. ":dynamic")
end, false)

RegisterNetEvent("duel:" .. GetCurrentResourceName() .. ":dynamic", function()
    print("never a symbol")
end)

MySQL.query("duel:both", {}, function() end)
