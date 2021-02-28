# encoding: utf-8

from cloudsml_computational_backend_common.analysis.consts import PredictiveModelExportFormats

from .base import str_to, BaseTranslator


TRANSLATION_MAP = {
        PredictiveModelExportFormats.python: {
            'if': '',
            'new': '',
            'ifnotnull': '',
            'a.sum': '',
            'a.argmax': (
                    '\n'
                    '\n'
                    'def argmax(arg):\n'
                    '    return numpy.argmax(numpy.array(arg))'
                ),
            'la.add': (
                    '\n'
                    '\n'
                    'def add(first_arg, second_arg):\n'
                    '    first_arg = numpy.array(first_arg)\n'
                    '    second_arg = numpy.array(second_arg)\n'
                    '    if(\n'
                    '          (len(first_arg.shape) + len(second_arg.shape)) == 3\n'
                    '          and (first_arg.shape[0] == second_arg.shape[0])\n'
                    '    ):\n'
                    '        first_arg = first_arg.T\n'
                    '    result = numpy.add(first_arg, second_arg)\n'
                    '    if len(result.shape) == 2:\n'
                    '        result = result.T\n'
                    '    return result.tolist()'
                ),
            'la.dot': (
                    '\n'
                    '\n'
                    'def dot(first_arg, second_arg):\n'
                    '    return numpy.dot(\n'
                    '           numpy.array(first_arg),\n'
                    '           numpy.array(second_arg)\n'
                    '        ).tolist()'
                ),
            'm.kernel.linear': (
                    '\n'
                    '\n'
                    'def linear(first_arg, second_arg):\n'
                    '    return numpy.dot(\n'
                    '           numpy.array(first_arg),\n'
                    '           numpy.array(second_arg)\n'
                    '       ).tolist()'
                ),
            'm.exp': (
                    '\n'
                    '\n'
                    'def exp(arg):\n'
                    '    return numpy.exp(numpy.array(arg)).tolist()'
                ),
            'm.link.logit': (
                    '\n'
                    '\n'
                    'def logit(arg):\n'
                    '    return (1.0 / (1.0 + numpy.exp(-numpy.array(arg)))).tolist()'
                ),
            'head': (
                    'import math\n'
                    'import numpy\n'
                    'import sys\n'
                    'import collections\n'
                    'import csv\n'
                    '\n'
                    '%s\n'
                ),
            'str_to_function': (
                    '\n'
                    '\n'
                    'def str_to(value):\n'
                    '    if isinstance(value, str):\n'
                    '         try:\n'
                    '             if \'.\' not in value:\n'
                    '                 return int(value)\n'
                    '             else:\n'
                    '                 return float(value)\n'
                    '         except ValueError:\n'
                    '             return math.nan\n'
                    '    else:\n'
                    '         return math.nan'
                ),
            'action': (
                    '\n'
                    '\n'
                    'def action(input):\n'
                    '    return %s'
                ),
            'run': (
                    '\n'
                    '\n'
                    'def run():\n'
                    '    data_file_name = sys.argv[1]\n'
                    '    data_file = open(data_file_name, \'r\')\n'
                    '    data_source = csv.reader(data_file)\n'
                    '    predictions = []\n'
                    '    data_source.__next__()\n'
                    '    for record in data_source:\n'
                    '        record = [str_to(item) for item in record]\n'
                    '        predictions.append(action(Record(*record[:-1])))\n'
                    '    print(predictions)\n'
                    '    data_file.close()'
                    '\n'
                    '\n'
                    'if __name__ == \'__main__\':\n'
                    '    run()'
                ),
            'input_data_struct': (
                    lambda fields: '\n\n'
                        'Record = '
                        'collections.namedtuple(\'Record\', [%s])' % \
                        ', '.join(
                            '\'%s\'' % record_field_description['name']\
                                for record_field_description in fields
                            )
                ),
            'binary_operators_mapping': {
                    '+': '+',
                    '-': '-',
                    '*': '*',
                    '/': '/',
                    '==': '==',
                    '!=': '!=',
                    '<': '<',
                    '<=': '<=',
                    '>': '>',
                    '>=': '>='
                }
            },
        'Cython': {},
        PredictiveModelExportFormats.c: {
            'if': '',
            'new': (
                    '\n'
                    '\n'
                    'struct Matrix matrix_new(\n'
                    '        size_t rownum,\n'
                    '        size_t colnum, ...) {\n'
                    '    size_t i, j;\n'
                    '    struct Matrix A;\n'
                    '    va_list argptr;\n'
                    '    va_start(argptr, colnum);\n'
                    '    A = create_matrix(rownum, colnum);\n'
                    '    for (i = 0; i < rownum; i++) {\n'
                    '        for (j = 0; j < colnum; j++) {\n'
                    '            A.elements[i][j] = va_arg(argptr, double);\n'
                    '        }\n'
                    '    }\n'
                    '    va_end(argptr);\n'
                    '    return A;\n'
                    '}'
                ),
            'ifnotnull': '',
            'a.sum': (
                    '\n'
                    '\n'
                    'float sum(struct Matrix A) {\n'
                    '    size_t i, j;\n'
                    '    float result;\n'
                    '    result = 0;\n'
                    '    for (i = 0; i < A.rownum; i++) {\n'
                    '        for (j = 0; j < A.colnum; j++) {\n'
                    '            result = result + A.elements[i][j];\n'
                    '        }\n'
                    '    }\n'
                    '    destroy_matrix(A);\n'
                    '    return result;\n'
                    '}'
                ),
            'a.argmax': (
                    '\n'
                    '\n'
                    'size_t argmax(struct Matrix A) {\n'
                    '    size_t i, i_max = 0;\n'
                    '    float max = 0.0;\n'
                    '    if (A.rownum == 1) {\n'
                    '        for (i = 0; i < A.colnum; i++) {\n'
                    '            if (A.elements[0][i] > max) {\n'
                    '                max = A.elements[0][i];\n'
                    '                i_max = i;\n'
                    '            }\n'
                    '        }\n'
                    '    }\n'
                    '    else if (A.colnum == 1) {\n'
                    '        for (i = 0; i < A.rownum; i++) {\n'
                    '            if (A.elements[i][0] > max) {\n'
                    '                max = A.elements[i][0];\n'
                    '                i_max = i;\n'
                    '            }\n'
                    '        }\n'
                    '    }\n'
                    '    else {\n'
                    '        return UINT_MAX;\n'
                    '    }\n'
                    '    destroy_matrix(A);\n'
                    '    return i_max;\n'
                    '}'
                ),
            'la.add': (
                    '\n'
                    '\n'
                    'struct Matrix add(struct Matrix X, struct Matrix Y) {\n'
                    '    struct Matrix result;\n'
                    '    size_t i, j;\n'
                    '    if (X.rownum == Y.colnum && X.colnum == Y.rownum) {\n'
                    '        Y = transpose(Y);\n'
                    '    }'
                    '    else if (X.rownum != Y.rownum || X.colnum != Y.colnum) {\n'
                    '        result.rownum = 0;\n'
                    '        result.colnum = 0;\n'
                    '        return result;\n'
                    '    }\n'
                    '    result = create_matrix(X.rownum, X.colnum);\n'
                    '    for (i = 0; i < X.rownum; i++) {\n'
                    '        for (j = 0; j < X.colnum; j++) {\n'
                    '            result.elements[i][j] = X.elements[i][j] + Y.elements[i][j];\n'
                    '        }\n'
                    '    }\n'
                    '    return result;\n'
                    '}'
                ),
            'la.dot': (
                    '\n'
                    '\n'
                    'struct Matrix dot(struct Matrix X, struct Matrix Y) {\n'
                    '    struct Matrix result;\n'
                    '    size_t i, j, k;\n'
                    '    if (X.colnum != Y.rownum) {\n'
                    '        result.rownum = 0;\n'
                    '        result.colnum = 0;\n'
                    '        return result;\n'
                    '    }\n'
                    '    result = create_matrix(X.rownum, Y.colnum);\n'
                    '    for (i = 0; i < X.rownum; i++) {\n'
                    '        for (j = 0; j < Y.colnum; j++) {\n'
                    '            result.elements[i][j] = 0;\n'
                    '            for (k = 0; k < X.colnum; k++) {\n'
                    '                result.elements[i][j] = result.elements[i][j] + X.elements[i][k] * Y.elements[k][j];\n'
                    '            }\n'
                    '        }\n'
                    '    }\n'
                    '    destroy_matrix(X);\n'
                    '    destroy_matrix(Y);\n'
                    '    return result;\n'
                    '}'
                ),
            'm.kernel.linear': (
                    '\n'
                    '\n'
                    'float linear(struct Matrix X, struct Matrix Y) {\n'
                    '    float result = 0;\n'
                    '    size_t i;\n'
                    '    for (i = 0; i < X.colnum; i++) {\n'
                    '        result = result + X.elements[0][i] * Y.elements[0][i];\n'
                    '    }\n'
                    '    destroy_matrix(X);\n'
                    '    destroy_matrix(Y);\n'
                    '    return result;\n'
                    '}'
                ),
            'm.exp': '',
            'm.link.logit': (
                    '\n'
                    '\n'
                    'struct Matrix logit(struct Matrix A) {\n'
                    '    size_t i, j;\n'
                    '    struct Matrix result = create_matrix(A.rownum, A.colnum);\n'
                    '    for (i = 0; i < A.rownum; i++) {\n'
                    '        for (j = 0; j < A.colnum; j++) {\n'
                    '            result.elements[i][j] = 1.0 / (1.0 + exp(-A.elements[i][j]));\n'
                    '        }\n'
                    '    }\n'
                    '    destroy_matrix(A);\n'
                    '    return result;\n'
                    '}'
                ),
            'head': (
                    '#include <stdio.h>\n'
                    '#include <stdlib.h>\n'
                    '#include <limits.h>\n'
                    '#include <math.h>\n'
                    '#include <stdarg.h>\n'
                    '#include <stdbool.h>\n'
                    '\n'
                    '%s\n'
                    '\n'
                    'struct Matrix {\n'
                    '    size_t rownum;\n'
                    '    size_t colnum;\n'
                    '    float** elements;\n'
                    '};\n'
                    '\n'
                    'struct Matrix create_matrix(\n'
                    '    size_t rownum,\n'
                    '    size_t colnum) {\n'
                    '    struct Matrix A;\n'
                    '    size_t i;\n'
                    '    if (rownum == 0 || colnum == 0) {\n'
                    '        A.rownum = 0;\n'
                    '        A.colnum = 0;\n'
                    '        return A;\n'
                    '    }\n'
                    '    A.rownum = rownum;\n'
                    '    A.colnum = colnum;\n'
                    '    A.elements = (float**)malloc(rownum * sizeof(float*));\n'
                    '    for (i = 0; i < rownum; i++) {\n'
                    '        A.elements[i] = (float*)malloc(colnum * sizeof(float));\n'
                    '    }\n'
                    '    return A;\n'
                    '}'
                    '\n'
                    '\n'
                    'void destroy_matrix(struct Matrix A) {\n'
                    '    size_t i;\n'
                    '    for (i = 0; i < A.rownum; i++) {\n'
                    '        free((void*)A.elements[i]);\n'
                    '    }\n'
                    '    free((void**)A.elements);\n'
                    '}'
                    '\n'
                    '\n'
                    'struct Matrix transpose(struct Matrix A) {\n'
                    '    size_t i, j;\n'
                    '    struct Matrix T = create_matrix(A.colnum, A.rownum);\n'
                    '    for (i = 0; i < A.rownum; i++) {\n'
                    '        for (j = 0; j < A.colnum; j++) {\n'
                    '            T.elements[j][i] = A.elements[i][j];\n'
                    '        }\n'
                    '    }\n'
                    '    destroy_matrix(A);\n'
                    '    return T;\n'
                    '}'
                ),
            'str_to_function': '',
            'action': (
                    '\n'
                    '\n'
                    'float action(const struct Record* input) {\n'
                    '    return %s;\n'
                    '}'
                ),
            'run': (
                    '\n'
                    '\n'
                    'int main(int argc, char* argv[]) {\n'
                    '    float temp;\n'
                    '    struct Record input;\n'
                    '    FILE *fp;\n'
                    '    fp = fopen(argv[1], \"r\");\n'
                    '    while(fgetc(fp) != \'\\n\') {\n'
                    '        ;\n'
                    '    }'
                    '    while(fscanf(fp, "%s\\n", &input.%s, &temp) != EOF) {\n'
                    '        printf(\"%s,", action(&input));\n'
                    '    }\n'
                    '    fclose(fp);\n'
                    '    return 0;\n'
                    '}'
                ),
            'input_data_struct': (
                    lambda fields, types_mapping: '\n\n'
                        'struct Record {\n%s\n};' % \
                        '\n'.join(
                            '    %s %s;' % (
                                    types_mapping[
                                       cast_pfa_type_to_c(record_field_description['type'])
                                        ],
                                record_field_description['name']
                                ) for record_field_description in fields
                            )
                ),
            'binary_operators_mapping': {
                    '+': '+',
                    '-': '-',
                    '*': '*',
                    '/': '/',
                    '==': '==',
                    '!=': '!=',
                    '<': '<',
                    '<=': '<=',
                    '>': '>',
                    '>=': '>='
                },
            'types_mapping': {
                    'integer': 'int',
                    'long': 'long int',
                    'float': 'float',
                    'double': 'float',
                    'string': '*char'
                },
            'types_format_mapping': {
                    'integer': '%d',
                    'long': '%d',
                    'float': '%f',
                    'double': '%f',
                    'string': '%s'
                }
            },
        PredictiveModelExportFormats.java: {}
        }


def cast_pfa_type_to_c(type_exression):
    if isinstance(type_exression, str):
        return type_exression
    elif (
            isinstance(type_exression, list) and
            ((type_exression == ['double', 'null']) or (type_exression == ['null', 'double']))):
        return 'double'
    else:
        raise ValueError('Unknown PFA type %s' % type_exression)


class Translator(BaseTranslator):
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
        super(self.__class__, self).__init__(pfa_document=pfa_document)
        self.translated_built_in_names = set()
        self.lang = None

    def translate_pfa_document(self, lang, output_file=None, pfa_document=None):
        if lang in TRANSLATION_MAP:
            self.lang = lang
        else:
            raise ValueError("Unknown translation language")

        lang_translation_map = TRANSLATION_MAP[self.lang]

        if output_file is None:
            raise ValueError("Output file not defined")
        if pfa_document is not None:
            self.pfa_document = pfa_document
        if self.pfa_document is None:
            raise ValueError("PFA document not defined")
        if self.check_document_correctness() is False:
            raise ValueError("PFA document is corrupted! Stop translation!")
        input_description = self.pfa_document['input']
        input_description_type = self.get_input_description_type(input_description)

        if self.recognise_record_input_description(input_description_type):
            if self.lang is PredictiveModelExportFormats.python:
                input_data_struct = lang_translation_map['input_data_struct'](
                        input_description['fields']
                    )
            elif self.lang is PredictiveModelExportFormats.c:
                input_data_struct = lang_translation_map['input_data_struct'](
                        input_description['fields'],
                        lang_translation_map['types_mapping']
                    )
            else:
                raise NotImplementedError(
                        "Input record format not defined for language %s" % (self.lang,)
                    )
            output_file.write(lang_translation_map['head'] % input_data_struct)
        output_file.write(lang_translation_map['str_to_function'])
        self.translated_built_in_names = set()
        action = lang_translation_map['action'] % self.translate(pfa_document['action'])
        if self.lang == PredictiveModelExportFormats.c:
            action = action.replace('input.', 'input->')
        for builtin_name in sorted(self.translated_built_in_names):
            output_file.write(lang_translation_map[builtin_name])
        output_file.write(action)
        if self.lang == PredictiveModelExportFormats.python:
            output_file.write(lang_translation_map['run'])
        elif self.lang == PredictiveModelExportFormats.c:
            fields = input_description['fields']
            types_format_mapping = lang_translation_map['types_format_mapping']
            output_file.write(
                    lang_translation_map['run'] % \
                    (
                        ','.join(types_format_mapping[
                                cast_pfa_type_to_c(record_field_description['type'])
                            ]
                            for record_field_description in fields
                        ) + ',%f',
                        ', &input.'.join(record_field_description['name']
                            for record_field_description in fields
                        ),
                        '%f',
                    )
                )
        output_file.flush()

    def translate_explicit_type_declaration(self, current_item):
        value = list(current_item.values())[0]
        if self.detect_string(value):
            return str_to(value)
        return value

    def translate_literal(self, current_item):
        if (
                self.detect_boolean(current_item) or
                self.detect_integer(current_item) or
                self.detect_float(current_item)
                ):
            return str(current_item)
        elif self.detect_string(current_item):
            return current_item
        elif self.detect_bracketed_string(current_item):
            return current_item[0]
        elif self.detect_explicit_type_declaration(current_item):
            return self.translate_explicit_type_declaration(current_item)
        elif self.detect_type_value_special_form(current_item):
            return self.translate(current_item['value'])
        raise ValueError("'%s' is a wrong literal" % (current_item, ))

    def detect_expression(self, current_item):
        return (
                isinstance(current_item, dict) or
                self.detect_literal(current_item)
            )

    def translate_binary_op(self, current_expression):
        # TODO List with binary operation arguments is expressions list.
        # In future when will be realise interpret_expressions_list
        # deal with bin op arguments list as with expressions list
        # But, may be arguments list interpeted in special way, not
        # as expressions list. Binary op consume all argumetns values
        # but, if iterpret arguments list as expressions list will be
        # return only last (last in list) expression value.
        binary_op_translation_map = TRANSLATION_MAP[self.lang][
                'binary_operators_mapping'
            ]
        binary_op = list(current_expression.keys())[0]
        first_term, second_term = current_expression[binary_op]
        first_term = self.translate(first_term)
        second_term = self.translate(second_term)
        if binary_op not in binary_op_translation_map.keys():
            raise NotImplementedError("Unknown binary orperation %s", (binary_op, ))
        else:
            return "(%s %s %s)" % (
                    first_term,
                    binary_op_translation_map[binary_op],
                    second_term
                )

    def translate_exponent(self, current_item):
        self.translated_built_in_names.add('m.exp')
        return 'exp(%s)' % self.translate(current_item['m.exp'][0])

    def translate_linear(self, current_item):
        self.translated_built_in_names.add('m.kernel.linear')
        first_arg, second_arg = current_item['m.kernel.linear']
        return "linear(%s, %s)" % (
                self.translate(first_arg),
                self.translate(second_arg)
            )

    def translate_logit(self, current_item):
        self.translated_built_in_names.add('m.link.logit')
        return "logit(%s)" % self.translate(current_item['m.link.logit'])

    def translate_array_argmax(self, current_item):
        self.translated_built_in_names.add('a.argmax')
        return 'argmax(%s)' % self.translate(current_item['a.argmax'])

    def translate_math_function(self, current_item):
        if self.detect_exponent(current_item):
            return self.translate_exponent(current_item)
        if self.detect_linear(current_item):
            return self.translate_linear(current_item)
        if self.detect_transpose(current_item):
            return self.translate_transpose(current_item)
        if self.detect_la_dot(current_item):
            return self.translate_la_dot(current_item)
        if self.detect_la_add(current_item):
            return self.translate_la_add(current_item)
        if self.detect_logit(current_item):
            return self.translate_logit(current_item)
        raise NotImplementedError("Unknown math function")

    def translate_if_form(self, current_item):
        if self.lang == PredictiveModelExportFormats.python:
            return "((%s) if (%s) else (%s))" % (
                    self.translate(current_item['then']),
                    self.translate(current_item['if']),
                    self.translate(current_item['else'])
                )
        elif self.lang == PredictiveModelExportFormats.c:
            return "((%s) ? (%s) : (%s))" % (
                    self.translate(current_item['if']),
                    self.translate(current_item['then']),
                    self.translate(current_item['else'])
                )
        else:
            raise NotImplementedError(
                    "if special form translation not implemented for language %s" % (self.lang,)
                )

    def translate_new_form(self, current_item):
        self.translated_built_in_names.add('new')
        if self.lang is PredictiveModelExportFormats.python:
            return "[%s]" % ', '.join(
                    self.translate(item) for item in current_item['new']
                )
        elif self.lang is PredictiveModelExportFormats.c:
            new_data = current_item['new']
            if (not isinstance(new_data[0], dict)) or ('new' not in new_data[0]):
                return "matrix_new(1, %d, %s)" % (
                        len(new_data),
                        ', '.join(self.translate(item) for item in new_data)
                    )
            elif 'new' in new_data[0]:
                rownum = len(new_data)
                colnum = len(new_data[0]['new'])
                data = []
                for item in new_data:
                    data.extend(item['new'])
                return "matrix_new(%d, %d, %s)" % (
                        rownum,
                        colnum,
                        ', '.join(self.translate(item) for item in data)
                    )
            else:
                raise ValueError("Unknown PFA array element type: %s" % new_data[0])
        else:
            raise NotImplementedError(
                    "Translation for new form not implemented for language %s" % \
                            (self.lang, )
                )

    def translate_ifnotnull_form(self, current_item):
        self.translated_built_in_names.add('ifnotnull')
        if self.lang is PredictiveModelExportFormats.python:
            return "((%s) if not (%s) else (%s))" % (
                    self.translate(current_item['then']),
                    " or ".join(
                            "math.isnan(%s)" %
                                self.translate(item) for item in current_item['ifnotnull'].values()
                        ),
                    self.translate(current_item['else'])
                )
        elif self.lang is PredictiveModelExportFormats.c:
            potentially_nans = current_item['ifnotnull'].values()
            return "(!(%s) ? (%s) : (%s))" % (
                    ' || '.join(
                        'isnan(%s)' % self.translate(item) for item in potentially_nans
                        ),
                    self.translate(current_item['then']),
                    self.translate(current_item['else'])
                )
        else:
            raise NotImplementedError(
                    "Translation for ifnotnull form not implemented for language %s" % \
                            (self.lang, )
                )

    def translate_special_form(self, current_item):
        if self.detect_if_form(current_item):
            return self.translate_if_form(current_item)
        if self.detect_new_form(current_item):
            return self.translate_new_form(current_item)
        if self.detect_ifnotnull_form(current_item):
            return self.translate_ifnotnull_form(current_item)
        raise NotImplementedError(
                "special form \"%s\" is not supported." % current_item
            )

    def translate_array_sum(self, current_item):
        self.translated_built_in_names.add('a.sum')
        return "sum(%s)" % self.translate_new_form(current_item['a.sum'])

    def translate_la_add(self, current_item):
        self.translated_built_in_names.add('la.add')
        first_arg, second_arg = current_item['la.add']
        return "add(%s, %s)" % (self.translate(first_arg), self.translate(second_arg))

    def translate_la_dot(self, current_item):
        self.translated_built_in_names.add('la.dot')
        first_arg, second_arg = current_item['la.dot']
        return "dot(%s, %s)" % (self.translate(first_arg), self.translate(second_arg))

    def translate_array_op(self, current_item):
        if self.detect_array_sum(current_item):
            return self.translate_array_sum(current_item)
        elif self.detect_array_zipmap(current_item):
            return self.translate_array_zipmap(current_item)
        elif self.detect_array_argmax(current_item):
            return self.translate_array_argmax(current_item)
        raise NotImplementedError(
                "array op \"%s\" is not supported." % current_item
            )

    def translate(self, current_item):
        """
        Args:
            current_item (JSON as python structure):
            Current PFA lexical unit retrieving by processing ("parsing")
            PFA engine JSON.

        Returns:
            (string): Soruce code in target programming language.
        """
        translation_result = None
        if self.detect_expression(current_item):
            if self.detect_literal(current_item):
                translation_result = self.translate_literal(current_item)
            elif self.detect_binary_op(current_item):
                translation_result = self.translate_binary_op(current_item)
            elif self.detect_special_form(current_item):
                translation_result = self.translate_special_form(current_item)
            elif self.detect_array_op(current_item):
                translation_result = self.translate_array_op(current_item)
            elif self.detect_math_function(current_item):
                translation_result = self.translate_math_function(current_item)
            else:
                raise NotImplementedError(
                        "'%s' cannot be translated yet" % (current_item,)
                    )
        return translation_result
