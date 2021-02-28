# encoding: utf-8

"""
Encoder for missing values
--------------------------
"""

FILL_MISSING_CONST = 9e+30


def missing_values_encoder(dask_df, columns_info):
    missing_values_substitution_map = {
            column_id: FILL_MISSING_CONST
                for column_id, column_info in columns_info.items() if
                    column_info.statistics['missing_values_count'] > 0
        }
    return dask_df.fillna(missing_values_substitution_map), missing_values_substitution_map
