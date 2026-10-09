(function_call name: (identifier) @ref.name) @ref.call
(function_call
  name: (dot_index_expression
    table: [(identifier) (dot_index_expression)] @ref.qualifier
    field: (identifier) @ref.name)) @ref.call
(function_call
  name: (method_index_expression
    table: (identifier) @ref.self
    method: (identifier) @ref.name)
  (#eq? @ref.self "self")) @ref.call
(function_call
  name: (method_index_expression
    table: [(identifier) (dot_index_expression)] @ref.qualifier
    method: (identifier) @ref.name)) @ref.call

(string content: (string_content) @ref.name) @ref.string

(function_call
  name: (identifier) @ref.qualifier
  arguments: (arguments . (string content: (string_content) @ref.name))) @ref.emit
(function_call
  name: (identifier) @ref.qualifier
  arguments: (arguments
    . (_) @ref.first
    . (string content: (string_content) @ref.name))
  (#not-match? @ref.first "^[\"']")) @ref.emit
