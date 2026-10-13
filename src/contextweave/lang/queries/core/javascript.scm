(function_declaration name: (identifier) @name) @def.function
(generator_function_declaration name: (identifier) @name) @def.function
(class_declaration name: (identifier) @name) @def.class
(method_definition name: (property_identifier) @name) @def.method

(variable_declarator name: (identifier) @name value: (arrow_function)) @def.function
(variable_declarator name: (identifier) @name value: (function_expression)) @def.function

(import_statement) @import.import
(export_statement source: (string) @import.module) @import.reexport
(call_expression
  function: (import)
  arguments: (arguments (string) @import.module)) @import.dynamic
(call_expression
  function: (identifier) @import.callee
  arguments: (arguments (string) @import.module)) @import.require

; FiveM/CFX JavaScript handlers bound to a string, the twin of the Lua rule.
; Same in node16/node22 (legacy), node26 (enhanced) and client V8:
;   on("name", () => {}), onNet("name", function () {}), AddEventHandler(...),
;   RegisterCommand("name", ...), exports("name", ...)
; The callee is pinned to a plain identifier, so `emitter.on("x", fn)` and
; `socket.on("x", fn)` stay out. The string is the only name the function has.
; A concatenated or templated name is dynamic and is not matched.
(call_expression
  function: (identifier) @handler.registrar
  arguments: (arguments
    . (string . (string_fragment) @name .)
    .
    [(function_expression) (arrow_function)])
  (#any-of? @handler.registrar
    "on" "onNet" "AddEventHandler" "RegisterNetEvent" "RegisterServerEvent"
    "addEventListener" "addNetEventListener" "RegisterCommand" "exports")) @def.handler
