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
    if self.is_valid_coeffs(coeffs): 
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

  def is_valid_coeffs(self, coeffs):
    is_valid = PolynomialData.__is_valid_coeffs_datatype(coeffs)

    if not is_valid:
      return False
    
    is_sparse = PolynomialData.__maybe_sparse(coeffs)

    if is_sparse and is_valid:
      for exp, coeff in coeffs.items():
        if not isinstance(exp, int) or not isinstance(coeff, Number):
          is_valid = False
    
    if is_sparse and is_valid:
      self._is_sparse = is_sparse
      return True

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
  
  @property
  def is_sparse(self):
    return self._is_sparse

class Polynomial():
  def __init__(self, coeffs: np.ndarray | dict, var: str = 'x'):
    self.__data = PolynomialData(coeffs)
    self._coefficients = self.__data.coefficients
    self._exponents = self.__data.exponents
    self._is_sparse = self.__data.is_sparse
    self._variable = var
  
  def __str__(self):
    vect_txt = np.vectorize(str)
    txt = list(vect_txt(self._coefficients))

    for i in range(len(self)):
      if self._coefficients[i] == 0:
        continue
      if self._exponents[i] == 0:
        txt[i] = f"{str(txt[i])}"

      if self._exponents[i] == 1:
        txt[i] = f"{str(txt[i])}{self._variable}"

      else:
        txt[i] = f"{str(txt[i])}{self._variable}^{self._exponents[i]}"

    txt = " + ".join(txt)
    
    return txt

  @staticmethod
  def __safe_dict_add(pairs1: dict, pairs2: dict, i: int, initial_sum: Number = 0):
    _sum = initial_sum
    if i in pairs1.keys():
      _sum += pairs1[i]
    if i in pairs2.keys():
      _sum += pairs2[i]
    return _sum

  def __add__(self, p):
    if isinstance(p, Polynomial):
      coef_exp_pairs_self = dict(zip(
        self._exponents,
        self._coefficients
      ))
      coef_exp_pairs_other = dict(zip(
        p.exponents,
        p.coefficients
      ))

      new_data = {}

      max_exp = max(max(self._exponents), max(p.exponents))

      for i in range(max_exp+1):
        new_data[i] = Polynomial.__safe_dict_add(
          coef_exp_pairs_self,
          coef_exp_pairs_other,
          i
        )

    if isinstance(p, Number):
      new_coeffs = self._coefficients[0] + p
      new_data = dict(zip(
        self._exponents,
        new_coeffs
      ))

    return Polynomial(new_data)

    
  def __mul__(self, p): pass
  def __eq__(self, p): pass
  def __len__(self):
    return len(self._coefficients)
  
  def evaluate(self, *x): pass
  def derivate(self): pass
  def integrate(self, *over): pass

  @property
  def coefficients(self):
    return self._coefficients

  @property
  def exponents(self):
    return self._exponents

  @property
  def is_sparse(self):
    return self._is_sparse


if __name__ == "__main__":
  p = Polynomial({9: 9})
  g = Polynomial({0: 9, 1: 2, 5: 6})
  print(p)
  print(g)
  print((p + g).coefficients, (p + g).exponents, p + g, sep="\n")

