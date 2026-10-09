RegisterNetEvent("duel:serverHello", function(count)
    local src = source
    TriggerClientEvent("duel:clientPing", src, count)
    TriggerClientEvent("duel:csClient", src, "hi")
    TriggerClientEvent("duel:both", src)
    TriggerEvent("duel:both")
    TriggerEvent("duel:sharedHook")
    TriggerLatentClientEvent("duel:clientPing", src, 5000)
end)

RegisterNetEvent("duel:both", function()
    print("server both")
end)

RegisterNetEvent("duel:bulk", function(data)
    print("bulk", data)
end)

RegisterNetEvent("duel:csBoth", function() end)
