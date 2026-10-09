(call function: (identifier) @ref.name) @ref.call
(call
  function: (attribute
    object: [(identifier) (attribute)] @ref.qualifier
    attribute: (identifier) @ref.name)) @ref.call
(call
  function: (attribute
    object: (identifier) @ref.self
    attribute: (identifier) @ref.name)
  (#any-of? @ref.self "self" "cls")) @ref.call

(class_definition superclasses: (argument_list (identifier) @ref.name)) @ref.inherit
(class_definition
  superclasses: (argument_list
    (attribute
      object: [(identifier) (attribute)] @ref.qualifier
      attribute: (identifier) @ref.name))) @ref.inherit

(string (string_content) @ref.name) @ref.string

(import_from_statement
  module_name: (relative_import (import_prefix) @ref.name)) @ref.relative
