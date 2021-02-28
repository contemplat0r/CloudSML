# encoding: utf-8
"""
Base implementations for PFA postprocessing
-------------------------------------------
"""


class BasePFATransformer(object):
    """
    It is a base implementation that should be inherited to actually modify
    PFA documents.
    """
    INPUT_STR_PATTERN = "input."
    INPUT_STR_PATTERN_LEN = len(INPUT_STR_PATTERN)

    def transform(self, pfa_document):
        """
        Walk through the ``pfa_document`` and apply some changes to the
        ``input`` and ``action`` sections.

        Args:
            pfa_document (dict): input PFA document.

        Returns:
            dict: a transformed ``pfa_document`` dict.
        """
        pfa_document['input']['fields'] = self._fields_transformer(pfa_document['input']['fields'])
        pfa_document['action'] = self._action_transformer(pfa_document['action'])
        return pfa_document

    def _fields_transformer(self, fields_descriptions):
        """
        Implement this method if you want to rename/replace input fields in the
        ``input`` section.

        Args:
            fields_descriptions (list):
                It is a list of dicts (with required ``name`` and ``type``
                keys). Learn more about ``input`` section in the PFA
                documentation.

        Returns:
            list: an updated ``fields_descriptions`` list.
        """
        return fields_descriptions

    def _column_reference_transformer(self, field_name):
        """
        Implement this method if you want to replace a reference to an input
        variable with some code or rename it.

        Args:
            field_name (str): name of a field which is referenced

        Returns:
            object: the output must be a valid PFA construction, so it can be
            either string reference to the column (``input.COLUMN_NAME``), or a
            dictionary with PFA instructions.
        """
        return self.INPUT_STR_PATTERN + field_name

    def _action_transformer(self, pfa_node):
        if isinstance(pfa_node, dict):
            return {
                    pfa_key: self._action_transformer(pfa_subnode) \
                        for pfa_key, pfa_subnode in pfa_node.items()
                }
        elif isinstance(pfa_node, list):
            return [self._action_transformer(pfa_subnode) for pfa_subnode in pfa_node]
        elif isinstance(pfa_node, str):
            if pfa_node.startswith(self.INPUT_STR_PATTERN):
                return self._column_reference_transformer(pfa_node[self.INPUT_STR_PATTERN_LEN:])
        return pfa_node
