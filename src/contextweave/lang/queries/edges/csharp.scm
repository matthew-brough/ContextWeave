(invocation_expression function: (identifier) @ref.name) @ref.call
(invocation_expression function: (generic_name (identifier) @ref.name)) @ref.call
(invocation_expression
  function: (member_access_expression
    expression: [(identifier) (member_access_expression)] @ref.qualifier
    name: (identifier) @ref.name)) @ref.call
(invocation_expression
  function: (member_access_expression
    "this" @ref.self
    name: (identifier) @ref.name)) @ref.call
(object_creation_expression type: (identifier) @ref.name) @ref.call
(object_creation_expression type: (generic_name (identifier) @ref.name)) @ref.call
(object_creation_expression
  type: (qualified_name
    qualifier: (_) @ref.qualifier
    name: (identifier) @ref.name)) @ref.call

(base_list (identifier) @ref.name) @ref.base
(base_list (generic_name (identifier) @ref.name)) @ref.base
(base_list
  (qualified_name
    qualifier: (_) @ref.qualifier
    name: (identifier) @ref.name)) @ref.base

(namespace_declaration name: (_) @scope.namespace)
(file_scoped_namespace_declaration name: (_) @scope.namespace)

(string_literal (string_literal_content) @ref.name) @ref.string

(invocation_expression
  function: (identifier) @ref.qualifier
  arguments: (argument_list
    . (argument (string_literal . (string_literal_content) @ref.name .)))) @ref.emit
(invocation_expression
  function: (identifier) @ref.qualifier
  arguments: (argument_list
    . (argument) @ref.first
    . (argument (string_literal . (string_literal_content) @ref.name .)))
  (#not-match? @ref.first "^[\"@$]")) @ref.emit
(invocation_expression
  function: (member_access_expression
    expression: (identifier) @_owner
    name: (identifier) @ref.qualifier)
  arguments: (argument_list
    . (argument (string_literal . (string_literal_content) @ref.name .)))
  (#eq? @_owner "BaseScript")) @ref.emit
(invocation_expression
  function: (member_access_expression
    expression: (identifier) @_owner
    name: (identifier) @ref.qualifier)
  arguments: (argument_list
    . (argument) @ref.first
    . (argument (string_literal . (string_literal_content) @ref.name .)))
  (#eq? @_owner "BaseScript")
  (#not-match? @ref.first "^[\"@$]")) @ref.emit
