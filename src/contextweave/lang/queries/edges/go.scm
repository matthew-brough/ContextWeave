(call_expression function: (identifier) @ref.name) @ref.call
(call_expression
  function: (selector_expression
    operand: (identifier) @ref.qualifier
    field: (field_identifier) @ref.name)) @ref.call

(struct_type
  (field_declaration_list
    (field_declaration !name type: (type_identifier) @ref.name) @ref.inherit))
(struct_type
  (field_declaration_list
    (field_declaration !name
      type: (pointer_type (type_identifier) @ref.name)) @ref.inherit))
(struct_type
  (field_declaration_list
    (field_declaration !name
      type: (qualified_type
        package: (package_identifier) @ref.qualifier
        name: (type_identifier) @ref.name)) @ref.inherit))
(interface_type (type_elem (type_identifier) @ref.name) @ref.implement)
(interface_type
  (type_elem
    (qualified_type
      package: (package_identifier) @ref.qualifier
      name: (type_identifier) @ref.name)) @ref.implement)

(interpreted_string_literal (interpreted_string_literal_content) @ref.name) @ref.string
