local Proxy = module("vrp", "lib/Proxy")
local Tunnel = module("vrp", "lib/Tunnel")
local json = require "json"
local helper = require("lib.helper")
local Other = notanimport("x")

RegisterNUICallback("closeMenu", function(data, cb)
  cb("ok")
end)

AddEventHandler("vrp:playerSpawn", function(user_id, source, first_spawn)
  return first_spawn
end)

RegisterCommand("vrpcmd", function(source, args)
  return args
end, false)

RegisterSecureServerEvent("vrp:secure", function()
  return 1
end)

exports("vrpExport", function()
  return 2
end)

MySQL.query("vrp/select", {}, function(rows)
  return rows
end)

AddEventHandler("vrp:" .. GetCurrentResourceName(), function()
  return 3
end)

Wait(500, function() end)
