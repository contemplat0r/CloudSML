import pytest

from app.modules.analysis.predictive_analysis import base


def test_model_build():
    with pytest.raises(NotImplementedError):
        base.BaseEstimator.build()

def test_bad_inheritance_base():
    """
    Ensure that we got TypeError raised on instantiating the inherited from abstract
    class if we don't declare required methods
    """
    class TestInheritBaseEstimator(base.BaseEstimator):
        def to_pfa(self):
            pass

    with pytest.raises(TypeError):
        instance = TestInheritBaseEstimator()
