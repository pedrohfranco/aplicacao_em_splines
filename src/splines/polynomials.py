import numpy as np
from numbers import Number
from typing import Tuple

"""
Abaixo, está a implementação de um polinômio univariado, com algumas classes auxiliares para a
geração do polinômio, assim como a implementação de algoritmos clássicos para a realização de
algumas operações matemáticas definidas sobre os polinômios.

A implementação utiliza de arranjos unidimensionais da biblioteca NumPy para a determinação dos
coeficientes e dos expoentes do polinômio. 

As operações matemáticas definidas, são:
- Adição e subtração;
- Multiplicação;
- Derivação e integração.
"""

class MathError(Exception):
  pass

class PolynomialData():
  def __init__(self, coeffs):
    if PolynomialData.is_valid_coeffs(coeffs): 
      self.c_array, self.e_array = self.__transform(coeffs)
    else:
      raise MathError(
        """Os coeficientes/expoentes informados não correspondem ao formato desejado.
           O argumento 'coeffs' deve ser de algum dos tipos: list, tuple, numpy.ndarray ou dict.
           Caso 'coeffs' for do tipo dict, os coeficientes devem ser inteiros.
           No geral, as componentes de 'coeffs' devem ser numéricas."""
        )

  @staticmethod
  def __is_valid_coeffs_datatype(coeffs):
    if PolynomialData.__maybe_sparse(coeffs):
      return True
    if isinstance(coeffs, (list, tuple, np.ndarray)):
      return True
    return False

  @staticmethod
  def __maybe_sparse(coeffs):
    if type(coeffs) == dict:
      return True
    return False

  @staticmethod
  def is_valid_coeffs(coeffs):
    is_valid = PolynomialData.__is_valid_coeffs_datatype(coeffs)

    if not is_valid:
      return False
    
    is_sparse = PolynomialData.__maybe_sparse(coeffs)

    if is_sparse and is_valid:
      for exp, coeff in coeffs.items():
        if not isinstance(exp, int) or not isinstance(coeff, Number):
          is_valid = False
    
    if is_sparse and is_valid: return True

    if is_valid:
      for coeff in coeffs:
        if not isinstance(coeff, Number):
          is_valid = False

    return is_valid

  def __transform(self, coeffs) -> Tuple[np.ndarray]:
    if PolynomialData.__maybe_sparse(coeffs):
      return (
        np.array(list(coeffs.keys())),
        np.array(list(coeffs.values()))
      )
    
    return (
      np.array(range(len(coeffs))),
      np.array(coeffs)
    )
  
  @property
  def exponents(self):
    return self.e_array

  @property
  def coefficients(self):
    return self.c_array

class Polynomial():
  def __init__(self, coeffs: np.ndarray | dict):
    self.__data = PolynomialData(coeffs)
    self._coefficients = self.__data.coefficients
    self._exponents = self.__data.exponents
  
  def __str__(self): pass
  def __add__(self, p): pass
  def __mul__(self, p): pass
  def __eq__(self, p): pass
  def evaluate(self, *x): pass
  def derivate(self): pass
  def integrate(self, *over): pass

  @property
  def coefficients(self):
    return self._coefficients

  @property
  def exponents(self):
    return self._exponents


if __name__ == "__main__":
  p = Polynomial({9.5: 9, 7: 5})
  print(p.exponents)
  print(p.coefficients)

