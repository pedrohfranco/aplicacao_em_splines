import numpy as np
from polynomials import Polynomial, MathError

"""
Os pontos imputados na classe do Interpolador por Splines devem estar no formato:
- dict: {x0: f(x0), x1: f(x1) ... xn: f(xn)} ou
- list: [[x0, f(x0)], [x1, f(x1)] ... [xn, f(xn)]]
Naturalmente, no input de uma lista, os pares x, f(x) podem estar em outro iterável ordenado.
"""

class CubicNaturalSpline(Polynomial):
  def __init__(self, x, *coeffs):
    binomial_term = Polynomial([-x, 1])
    binomial_term_sq = binomial_term * binomial_term
    binomial_term_cub = binomial_term_sq * binomial_term

    term_a = Polynomial([coeffs[0]])
    term_b = coeffs[1] * binomial_term
    term_c = coeffs[2] * binomial_term_sq
    term_d = coeffs[3] * binomial_term_cub
    
    spline = term_a + term_b + term_c + term_d

    super().__init__(spline.to_dense())
    self.x = x

class SplineInterpolator:
  def __init__(self, points: dict | list):
    if isinstance(points, dict):
      sorted_points = sorted(points.items())
    if isinstance(points, list):
      sorted_points = sorted(points)
    else:
      MathError("Os dados informados não estão no formato esperado. Os pontos devem ser pares" \
      "(x, f(x)) dos tipos dict ou list")
        
    self.x = np.array([p[0] for p in sorted_points], dtype=float)
    self.f = np.array([p[1] for p in sorted_points], dtype=float)
    self.n = len(self.x) - 1
    self.splines = []
    
    if self.n < 1:
        raise MathError("São necessários pelo menos 2 pontos para gerar um Spline.")
        
    self._compute_coefficients()

  def _compute_coefficients(self):
    h = np.diff(self.x)
    
    alpha = np.zeros(self.n)
    for i in range(1, self.n):
        term1 = (3 / h[i]) * (self.f[i+1] - self.f[i])
        term2 = (3 / h[i-1]) * (self.f[i] - self.f[i-1])
        alpha[i] = term1 - term2

    l = np.zeros(self.n + 1)
    mu = np.zeros(self.n + 1)
    z = np.zeros(self.n + 1)

    l[0] = 1.0

    for i in range(1, self.n):
        l[i] = 2 * (self.x[i+1] - self.x[i-1]) - h[i-1] * mu[i-1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i-1] * z[i-1]) / l[i]

    l[self.n] = 1.0
    
    c = np.zeros(self.n + 1)
    b = np.zeros(self.n)
    d = np.zeros(self.n)

    for j in range(self.n - 1, -1, -1):
        c[j] = z[j] - mu[j] * c[j+1]
        b[j] = (self.f[j+1] - self.f[j]) / h[j] - h[j] * (c[j+1] + 2 * c[j]) / 3
        d[j] = (c[j+1] - c[j]) / (3 * h[j])

    for i in range(self.n):
        spline = CubicNaturalSpline(
            self.x[i], self.f[i], b[i], c[i], d[i]
        )
        self.splines.append(spline)

  def evaluate(self, x):
    if x < self.x[0]:
        print("Warning: o ponto está fora do domínio do polinômio interpolador.")
        return self.splines[0].evaluate(x)
    
    if x > self.x[-1]:
        print("Warning: o ponto está fora do domínio do polinômio interpolador.")
        return self.splines[-1].evaluate(x)

    for i in range(self.n):
        if self.x[i] <= x <= self.x[i+1]:
            return self.splines[i].evaluate(x)
    
    return self.splines[-1].evaluate(x)

if __name__ == "__main__":
  pontos = {
      1: 2,
      2: 3,
      3: 5
  }
  
  interpolador = SplineInterpolator(pontos)
  
  print("Splines Gerados:")
  for i, s in enumerate(interpolador.splines):
      print(f"Intervalo [{interpolador.x[i]}, {interpolador.x[i+1]}]: {s}")
      
  x_teste = 1.5
  y_teste = interpolador.evaluate(x_teste)
  print(f"\nAvaliação em x={x_teste}: y={y_teste}")
  