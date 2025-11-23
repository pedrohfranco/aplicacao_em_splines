import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

"""
Abaixo está a implementação de um módulo genérico de plotagem 3D, que recebe os pontos
X,Y,Z ou recebe nós genéricos e permite: visualizar a superfície da função alvo,
aproximação ou erro, bem como os pontos de amostragem e as trajetórias de métodos
3D, quando aplicável.

"""

def create_3d_axes(title=None, xlabel="x", ylabel="y", zlabel="z"):
    """
    Cria uma figura e eixos 3D padronizados para o trabalho.
    Retorna (fig, ax).
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    if title:
        ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)

    return fig, ax


def finalize_figure(fig, ax, show=True, save_path=None, elev=25, azim=-60):
    """
    Ajusta ângulo de visão, salva e/ou mostra a figura.
    """
    ax.view_init(elev=elev, azim=azim)
    fig.tight_layout()

    if save_path is not None:
        folder = os.path.dirname(save_path)
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
        fig.savefig(save_path, dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_surface_xyz(X, Y, Z,
                     title=None,
                     xlabel="x",
                     ylabel="y",
                     zlabel="z",
                     cmap="viridis",
                     show=True,
                     save_path=None):
    """
    Plota uma superfície 3D genérica dada por matrizes X, Y, Z.
    Ideal para: f(x, y), erro(x, y), aproximações, etc.
    """
    fig, ax = create_3d_axes(title=title, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel)

    # surface
    surf = ax.plot_surface(X, Y, Z,
                           cmap=cmap,
                           linewidth=0,
                           antialiased=True)

    # barra de cores (opcional, mas bem útil p/ erro)
    fig.colorbar(surf, shrink=0.6, aspect=12, label=zlabel)

    finalize_figure(fig, ax, show=show, save_path=save_path)


def plot_surface_from_function(func,
                               x_range,
                               y_range,
                               nx=50,
                               ny=50,
                               title=None,
                               xlabel="x",
                               ylabel="y",
                               zlabel="z",
                               cmap="viridis",
                               show=True,
                               save_path=None):
    """
    Gera X, Y com meshgrid em [x_min, x_max] x [y_min, y_max],
    calcula Z = func(X, Y) e plota a superfície.

    func: função Python que recebe (X, Y) e devolve Z com mesma forma.
    """
    x_min, x_max = x_range
    y_min, y_max = y_range

    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)

    plot_surface_xyz(X, Y, Z,
                     title=title,
                     xlabel=xlabel,
                     ylabel=ylabel,
                     zlabel=zlabel,
                     cmap=cmap,
                     show=show,
                     save_path=save_path)


def plot_points_3d(x, y, z,
                   title=None,
                   xlabel="x",
                   ylabel="y",
                   zlabel="z",
                   show=True,
                   save_path=None):
    """
    Plota pontos 3D (amostras, nós, iterações, etc).
    """
    fig, ax = create_3d_axes(title=title, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel)

    ax.scatter(x, y, z, s=20)  # s = tamanho dos marcadores

    finalize_figure(fig, ax, show=show, save_path=save_path)


def plot_curve_3d(x, y, z,
                  title=None,
                  xlabel="x",
                  ylabel="y",
                  zlabel="z",
                  show=True,
                  save_path=None):
    """
    Plota uma curva 3D (útil pra mostrar trajetória de um método iterativo.
    """
    fig, ax = create_3d_axes(title=title, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel)

    ax.plot(x, y, z, marker="o")

    finalize_figure(fig, ax, show=show, save_path=save_path)
