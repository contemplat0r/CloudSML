# encoding: utf-8
from abc import ABCMeta, abstractmethod, abstractclassmethod

class BaseEstimator(metaclass=ABCMeta):

    @abstractclassmethod
    def build(self, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def to_pfa(self):
        raise NotImplementedError()

    @abstractmethod
    def get_info(self, input_dataset):
        raise NotImplementedError()
