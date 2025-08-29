import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def criar_grade(alt, larg):
    return np.zeros((alt, larg), dtype=int)

def colocar_aldeias(grade, posicoes, ids=None):
    if ids is None:
        ids = list(range(1, len(posicoes)+1))
    for (r, c), v in zip(posicoes, ids):
        grade[r, c] = v
    return grade

def mostrar_grade(grade):
    alt, larg = grade.shape

    cores = mcolors.ListedColormap(["lightgrey", "purple", "orange"])
    limites = [-0.5, 0.5, 1.5, 2.5]
    normalizacao = mcolors.BoundaryNorm(limites, cores.N)

    figura, eixo = plt.subplots()
    eixo.imshow(grade, cmap=cores, norm=normalizacao)

    # setar as linhas do grid
    eixo.set_xticks(np.arange(-0.5, larg, 1), minor=True)
    eixo.set_yticks(np.arange(-0.5, alt, 1), minor=True)

    eixo.grid(which="minor", color="black", linestyle="-", linewidth=0.5)

    # tira os números
    eixo.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

    plt.show()

# teste
alt, larg = 10, 15
g = criar_grade(alt, larg)
g = colocar_aldeias(g, [(2, 2), (7, 12)], ids=[1, 2])  
mostrar_grade(g)
