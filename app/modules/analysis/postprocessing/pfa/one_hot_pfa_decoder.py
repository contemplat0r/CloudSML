# encoding: utf-8

"""
PFA decoder for categorical values
----------------------------------
"""
from collections import namedtuple

from .base import BasePFATransformer


ColumnInfo = namedtuple('ColumnInfo', ['label', 'category_value'])


class OneHotPFADecoder(BasePFATransformer):
    """
    Reverse transformation one-hot encoded columns to categorical columns in PFA document
    """
    def __init__(self, categorical_columns):
        self.virtual_columns_mapping = {
                'var%s' % virtual_column.label: ColumnInfo(
                        label=column_label,
                        category_value=virtual_column.category_value
                    ) for column_label, virtual_columns in categorical_columns.items() \
                        for virtual_column in virtual_columns
            }

    def _fields_transformer(self, fields_descriptions):
        decoded_fields_descriptions = []
        decoded_fields_labels = set()
        for field_description in fields_descriptions:
            field_name = field_description['name']
            if field_name in self.virtual_columns_mapping:
                original_column = self.virtual_columns_mapping[field_name]
                if original_column.label in decoded_fields_labels:
                    continue
                decoded_fields_labels.add(original_column.label)
                field_description = {'name': 'var%s' % original_column.label, 'type': 'double'}
            decoded_fields_descriptions.append(field_description)
        return decoded_fields_descriptions

    def _column_reference_transformer(self, field_name):
        if field_name in self.virtual_columns_mapping:
            original_column = self.virtual_columns_mapping[field_name]
            return {
                    'if': {
                            '==': [
                                    '%svar%s' % (self.INPUT_STR_PATTERN, original_column.label),
                                    original_column.category_value
                                ]
                        },
                    'then': {'double': 1.0},
                    'else': {'double': 0.0}
                }
        return '%s%s' % (self.INPUT_STR_PATTERN, field_name)
