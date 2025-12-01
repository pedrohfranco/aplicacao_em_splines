import numpy as np
from numbers import Number
from typing import Tuple
import fft

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
      self.e_array, self.c_array = self.__transform(coeffs)
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
    self._is_sparse = False

    if not is_valid:
      return False
    
    is_sparse = PolynomialData.__maybe_sparse(coeffs)

    if is_sparse and is_valid:
      for exp, coeff in coeffs.items():
        if not isinstance(exp, Number) or not isinstance(coeff, (Number, complex)):
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
        np.array(list(coeffs.values()), dtype=complex)
      )
    
    return (
      np.array(range(len(coeffs))),
      np.array(coeffs, dtype=complex)
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
    if self.is_zero: return '0'

    vect_txt = np.vectorize(str)

    if ~self._coefficients.imag.all():
      str_coeffs = list(vect_txt(self._coefficients.real))
    else:
      str_coeffs = list(vect_txt(self._coefficients))

    txt = []

    for i in range(len(self)):
      if self._coefficients[i] == 0:
        continue
      if self._exponents[i] == 0:
        txt.append(f"{str_coeffs[i]}")
        continue
      if self._exponents[i] == 1:
        txt.append(f"{str_coeffs[i]}{self._variable}")
        continue
      else:
        txt.append(f"{str_coeffs[i]}{self._variable}^{self._exponents[i]}")

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
        _sum = Polynomial.__safe_dict_add(
          coef_exp_pairs_self,
          coef_exp_pairs_other,
          i
        )
        if _sum == 0: continue
        new_data[i] = _sum

    if isinstance(p, Number):
      new_coeffs = self._coefficients[0] + p
      new_data = dict(zip(
        self._exponents,
        new_coeffs
      ))

    return Polynomial(new_data)
  
  def __sub__(self, p):
    return self + -1 * p

  def to_dense(self, inplace: bool = False, return_poly: bool = False):
    if len(self) == 0:
      dense = np.array([0], dtype=complex)
      if inplace: self._coefficients = dense
    else:
      max_exp = np.max(self._exponents)
      dense = np.zeros(max_exp + 1, dtype=complex)
      dense[self._exponents] = self._coefficients
      if inplace: self._coefficients = dense
    return Polynomial(dense) if return_poly else dense

  def __mul__(self, p):
    if isinstance(p, Number):
      new_data = dict(zip(
        self._exponents,
        self._coefficients * p
      ))
      return Polynomial(new_data)

    if isinstance(p, Polynomial):
      dense_coeffs_self = self.to_dense()
      dense_coeffs_other = p.to_dense()

      final_coeffs = fft.evaluate(dense_coeffs_self, dense_coeffs_other)
      
      return Polynomial(final_coeffs)
      
    raise MathError("Multiplicação suportada apenas por escalares ou outros polinômios.")
  
  def __rmul__(self, p):
    return self.__mul__(p)

  def __eq__(self, p):
    if not np.array_equal(self._exponents, p.exponents): return False
    if not np.array_equal(self._coefficients, p.coefficients): return False
    return True

  def __len__(self):
    return len(self._coefficients)
  
  def evaluate(self, x):
    _inputs = np.pow(
      np.full(len(self), x),
      self._exponents)
    value = np.sum(np.multiply(self._coefficients, _inputs))

    return value

  def derivate(self, inplace: bool = False):
    if self.is_zero: return 0

    new_exp = self._exponents.copy()
    new_coeffs = self._coefficients.copy()

    if_less_zero = np.vectorize(lambda x: False if x == -1 else True)

    new_coeffs = np.multiply(new_coeffs, new_exp)
    new_exp -= 1

    mask = if_less_zero(new_exp)

    new_exp = new_exp[mask]
    new_coeffs = new_coeffs[mask]

    if inplace:
      self._exponents = new_exp
      self._coefficients = new_coeffs

    new_data = dict(zip(
      new_exp,
      new_coeffs
    ))
    if inplace:
      return None
    return Polynomial(new_data)

  def integrate(self, interval: list = [], inplace: bool = False):
    if self.is_zero: return 0

    new_exp = self._exponents.copy()
    new_coeffs = self._coefficients.copy()

    new_exp += 1

    new_coeffs = np.divide(new_coeffs, new_exp)

    if inplace:
      self._exponents = new_exp
      self._coefficients = new_coeffs

    new_data = dict(zip(
      new_exp,
      new_coeffs
    ))

    primitive = Polynomial(new_data)

    if 3 > len(interval) > 0:
      return primitive.evaluate(interval[1]) - primitive.evaluate(interval[0])

    if inplace:
      return None
    
    return primitive

  # Propriedades
  @property
  def coefficients(self):
    return self._coefficients

  @property
  def exponents(self):
    return self._exponents

  @property
  def is_zero(self):
    for boolean in self._coefficients == 0:
      if not boolean: return False
    return True

  @property
  def is_sparse(self):
    return self._is_sparse



if __name__ == "__main__":
  p = Polynomial({0: 9, 1: 2, 5: 6})
  g = Polynomial({0: 9, 1: 2})
  h = Polynomial([0, 0, 0, 0, 0])
  print(p * -1)
  print(p - g)
  print(p == g)
  print(h.evaluate(2))
  print(g.evaluate(2))
  print(p.derivate())

