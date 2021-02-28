# encoding: utf-8

"""
PFA Decoder for missing values
------------------------------
"""

from .base import BasePFATransformer


class MissingValuesPFADecoder(BasePFATransformer):
    """
    This PFA document transformer can be used to post-process the PFA operations
    which rely on columns which may contain missing values.
    """
    def __init__(self, substitution_map):
        self.substitution_map = substitution_map
        self.fields_with_missing = {
                'var%s' % column_id: column_id for column_id in self.substitution_map
            }

    def _fields_transformer(self, fields_descriptions):
        transformed_fields_descriptions = []
        for field_description in fields_descriptions:
            if field_description['name'] in self.fields_with_missing:
                field_description['type'] = ['double', 'null']
            transformed_fields_descriptions.append(field_description)
        return transformed_fields_descriptions

    def _column_reference_transformer(self, field_name):
        full_field_name = '%s%s' % (self.INPUT_STR_PATTERN, field_name)
        if field_name in self.fields_with_missing:
            return {
                    'ifnotnull': {full_field_name: full_field_name},
                    'then': full_field_name,
                    'else': self.substitution_map[self.fields_with_missing[field_name]]
                }
        return full_field_name
