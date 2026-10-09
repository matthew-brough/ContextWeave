(call_expression function: (identifier) @ref.name) @ref.call
(call_expression
  function: (member_expression
    object: [(identifier) (member_expression)] @ref.qualifier
    property: (property_identifier) @ref.name)) @ref.call
(call_expression
  function: (member_expression
    object: (this) @ref.self
    property: (property_identifier) @ref.name)) @ref.call
(new_expression constructor: (identifier) @ref.name) @ref.call
(new_expression
  constructor: (member_expression
    object: [(identifier) (member_expression)] @ref.qualifier
    property: (property_identifier) @ref.name)) @ref.call

(extends_clause value: (identifier) @ref.name) @ref.inherit
(extends_clause
  value: (member_expression
    object: [(identifier) (member_expression)] @ref.qualifier
    property: (property_identifier) @ref.name)) @ref.inherit
(extends_type_clause type: (type_identifier) @ref.name) @ref.inherit
(extends_type_clause
  type: (generic_type name: (type_identifier) @ref.name)) @ref.inherit

(implements_clause (type_identifier) @ref.name) @ref.implement
(implements_clause (generic_type name: (type_identifier) @ref.name)) @ref.implement
(implements_clause
  (nested_type_identifier
    module: (identifier) @ref.qualifier
    name: (type_identifier) @ref.name)) @ref.implement

(implements_clause
  (generic_type
    name: (nested_type_identifier
      module: (identifier) @ref.qualifier
      name: (type_identifier) @ref.name))) @ref.implement

(string (string_fragment) @ref.name) @ref.string

(call_expression
  function: (identifier) @ref.qualifier
  arguments: (arguments . (string . (string_fragment) @ref.name .))) @ref.emit
(call_expression
  function: (identifier) @ref.qualifier
  arguments: (arguments
    . (_) @ref.first
    . (string . (string_fragment) @ref.name .))
  (#not-match? @ref.first "^[\"'`]")) @ref.emit
