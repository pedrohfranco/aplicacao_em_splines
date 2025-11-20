import numpy as np

__all__ = ["evaluate"]

def fft(coeffs):
  n = len(coeffs)
  if n <= 1:
    return coeffs

  even = fft(coeffs[0::2])
  odd = fft(coeffs[1::2])

  factor = np.exp(-2j * np.pi * np.arange(n // 2) / n)

  return np.concatenate([even + factor * odd, even - factor * odd])

def ifft(coeffs):
  n = len(coeffs)
  return np.conjugate(fft(np.conjugate(coeffs))) / n

def evaluate(p1: np.ndarray, p2: np.ndarray):
  total_len = len(p1) + len(p2) - 1
  n = 1

  while n < total_len:
    n *= 2

  pad1 = np.pad(p1, (0, n - len(p1)))
  pad2 = np.pad(p2, (0, n - len(p2)))

  fft1 = fft(pad1)
  fft2 = fft(pad2)

  fft_result = fft1 * fft2
  result_dense = ifft(fft_result)
  result_real = np.round(result_dense.real, 10)

  return result_real[:total_len]