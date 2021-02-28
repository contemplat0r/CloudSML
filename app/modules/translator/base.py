# encoding: utf-8

import pandas

PRIMITIVE_AVRO_TYPES = {
        'boolean',
        'integer',
        'long',
        'float',
        'double',
        'string',
        'null',
        'base64'
    }

COMPOSITE_AVRO_TYPES = {
        'array',
        'map',
        'record',
        'enumeration_set',
        'byte_sequence',
        'fixed_width_byte_sequence',
        'tagged_union'
    }

AVRO_TYPES = PRIMITIVE_AVRO_TYPES | COMPOSITE_AVRO_TYPES

BINARY_OPERATIONS = {
        '+',
        '-',
        '*',
        '/',
        '<',
        '>',
        '<=',
        '>=',
        '=='
    }

SPECIAL_FORMS = {'if', 'new', 'ifnotnull'}

MATH_FUNCTIONS = {'m.exp', 'm.kernel.linear', 'm.link.logit'}

ARRAY_OPERATIONS = {'a.sum', 'a.zipmap', 'a.argmax'}

LINEAR_ALGEBRA_OPERATIONS = {'la.transpose', 'la.add', 'la.dot'}

BOOLEAN_SET = {'True', 'False'}


def str_to(value):
    if isinstance(value, str):
        result = ''
        try:
            if '.' not in value:
                result = int(value)
            else:
                result = float(value)
        except ValueError:
            result = value
        if value in BOOLEAN_SET:
            result = bool(result)
        return result
    else:
        return value


class BaseTranslator(object):
    """
    Attributes:
        pfa_document (JSON as complex python data structure):
            PFA document that describe model that interpreted
        SYMBOLS_REFERENCES (dict):
            Symbols reference (variables) table.
        input_names_variables_map (dict):
            Special symbols refrerence table for case is
            input data is record with special processing key names.
        input_names_types_map (dict):
            Special table for tracking variables types
            (for future used in to Cython translation)
        input_type (string):
            Input type that retrieve from PFA document that describe engine.
        prediction (list):
            field that contain all predictions after processing
            input data
    """

    def __init__(self, pfa_document=None):
        self.pfa_document = pfa_document

    def check_document_correctness(self):
        pfa_document = self.pfa_document
        if 'input' not in pfa_document:
            return False
        if 'output' not in pfa_document:
            return False
        if 'action' not in pfa_document:
            return False
        return True

    def recognise_primitive_input_description(self, input_description_type):
        return input_description_type in PRIMITIVE_AVRO_TYPES

    def recognise_array_input_description(self, input_description_type):
        return input_description_type == 'array'

    def recognise_record_input_description(self, input_description_type):
        return input_description_type == 'record'

    def get_input_description_type(self, input_description):
        input_description = self.pfa_document['input']
        if 'type' in input_description:
            return input_description['type']
        else:
            return input_description

    def detect_boolean(self, value):
        return isinstance(value, bool)

    def detect_integer(self, value):
        return isinstance(value, int)

    def detect_float(self, value):
        return isinstance(value, float)

    def detect_string(self, value):
        return isinstance(value, str)

    def detect_bracketed_string(self, value):
        return (
                isinstance(value, list) and
                len(value) == 1 and
                isinstance(value[0], str)
            )

    def detect_explicit_type_declaration(self, current_item):
        if not isinstance(current_item, dict):
            return False
        keys = list(current_item.keys())
        if len(keys) == 1:
            return keys[0] in PRIMITIVE_AVRO_TYPES
        return False

    def detect_type_value_special_form(self, current_item):
        return isinstance(current_item, dict)\
                and set(current_item.keys()) == {'type', 'value'}

    def detect_literal(self, current_item):
        return (
                self.detect_boolean(current_item) or
                self.detect_integer(current_item) or
                self.detect_float(current_item) or
                self.detect_string(current_item) or
                self.detect_bracketed_string(current_item) or
                self.detect_explicit_type_declaration(current_item) or
                self.detect_type_value_special_form(current_item)
            )

    def detect_symbol_reference(self, current_item):
        return (
                isinstance(current_item, str) and (
                    current_item in self.SYMBOLS_REFERENCES or
                    current_item in self.input_names_variables_map
                )
            )

    def detect_expression(self, current_item):
        return (
                isinstance(current_item, dict) or
                self.detect_literal(current_item) or
                self.detect_symbol_reference(current_item)
            )

    def detect_binary_op(self, current_item):
        return (
                isinstance(current_item, dict) and
                list(current_item.keys())[0] in BINARY_OPERATIONS
            )

    def detect_exponent(self, current_item):
        return list(current_item.keys())[0] == 'm.exp'

    def detect_linear(self, current_item):
        return list(current_item.keys())[0] == 'm.kernel.linear'

    def detect_logit(self, current_item):
        return list(current_item.keys())[0] == 'm.link.logit'

    def detect_array_argmax(self, current_item):
        return list(current_item.keys())[0] == 'a.argmax'

    def detect_math_function(self, current_item):
        return (
                isinstance(current_item, dict) and (
                    list(current_item.keys())[0] in MATH_FUNCTIONS or
                    list(current_item.keys())[0] in LINEAR_ALGEBRA_OPERATIONS
                )
            )

    def detect_if_form(self, current_item):
        return isinstance(current_item, dict) and 'if' in current_item

    def detect_new_form(self, current_item):
        return isinstance(current_item, dict) and 'new' in current_item

    def detect_ifnotnull_form(self, current_item):
        return isinstance(current_item, dict) and 'ifnotnull' in current_item

    def detect_special_form(self, current_item):
        return (
                isinstance(current_item, dict) and\
                    set(current_item.keys()) & set(SPECIAL_FORMS)
            )

    def detect_array_sum(self, current_item):
        return list(current_item.keys())[0] == 'a.sum'

    def detect_array_zipmap(self, current_item):
        return list(current_item.keys())[0] == 'a.zipmap'

    def detect_array_op(self, current_item):
        return (
                isinstance(current_item, dict) and
                (ARRAY_OPERATIONS & set(current_item.keys()))
            )

    def detect_transpose(self, current_item):
        return list(current_item.keys())[0] == 'la.transpose'

    def detect_la_add(self, current_item):
        return list(current_item.keys())[0] == 'la.add'

    def detect_la_dot(self, current_item):
        return list(current_item.keys())[0] == 'la.dot'

    def detect_interpretable(self, entity):
        return (
                self.detect_array_op(entity)\
                or
                self.detect_math_function(entity)\
                or
                self.detect_binary_op(entity)\
                or
                self.detect_symbol_reference(entity)
            )
