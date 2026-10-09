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
