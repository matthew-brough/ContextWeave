fx_version 'cerulean'
game 'gta5'

-- server_script 'server/commented.lua'
client_scripts {
    'client/*.lua',
    "client/Client.cs",
}

server_script 'server/main.lua'
server_script("server/Server.cs")

shared_scripts { 'shared.lua' }
