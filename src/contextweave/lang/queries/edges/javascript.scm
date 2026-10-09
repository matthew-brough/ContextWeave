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

(class_heritage (identifier) @ref.name) @ref.inherit
(class_heritage
  (member_expression
    object: [(identifier) (member_expression)] @ref.qualifier
    property: (property_identifier) @ref.name)) @ref.inherit

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
