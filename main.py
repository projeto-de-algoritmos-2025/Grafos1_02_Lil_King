import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from collections import deque  

def criar_grade(alt, larg):
    return np.zeros((alt, larg), dtype=int)

def configurar_cores_grade():
    cores = mcolors.ListedColormap(["lightgrey", "purple", "orange", "green"]) 
    limites = [-0.5, 0.5, 1.5, 2.5, 3.5]
    normalizacao = mcolors.BoundaryNorm(limites, cores.N)
    return cores, normalizacao

def configurar_grid_visual(eixo, altura, largura):
    eixo.set_xticks(np.arange(-0.5, largura, 1), minor=True)
    eixo.set_yticks(np.arange(-0.5, altura, 1), minor=True)
    eixo.grid(which="minor", color="black", linestyle="-", linewidth=0.5)
    eixo.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

def colocar_aldeias(grade, posicoes, ids=None):
    if ids is None:
        ids = list(range(1, len(posicoes)+1))
    for (r, c), v in zip(posicoes, ids):
        grade[r, c] = v
    return grade

def colocar_comidas(grade, posicoes):
    for (r, c) in posicoes:
        grade[r, c] = 3
    return grade

def mostrar_grade(grade):
    altura, largura = grade.shape
    cores, normalizacao = configurar_cores_grade()

    figura, eixo = plt.subplots()
    eixo.imshow(grade, cmap=cores, norm=normalizacao)

    configurar_grid_visual(eixo, altura, largura)
    plt.show()


def obter_vizinhos(linha, coluna, altura, largura):
    direcoes = [(-1,0), (1,0), (0,-1), (0,1)]
    
    for movimento_linha, movimento_coluna in direcoes:
        nova_linha = linha + movimento_linha
        nova_coluna = coluna + movimento_coluna
        
        if 0 <= nova_linha < altura and 0 <= nova_coluna < largura:
            yield (nova_linha, nova_coluna)

def encontrar_posicoes_com_valor(grade, valor):
    coordenadas = np.argwhere(grade == valor)
    return [tuple(coordenada) for coordenada in coordenadas]

def buscar_em_largura(grade, posicao_inicial):
    altura, largura = grade.shape
    
    fila = deque([posicao_inicial])
    visitados = {posicao_inicial}
    ordem_visita = []
    
    while fila:
        posicao_atual = fila.popleft()  
        ordem_visita.append(posicao_atual)
        
        for vizinho in obter_vizinhos(*posicao_atual, altura, largura):
            if vizinho not in visitados:
                visitados.add(vizinho)
                fila.append(vizinho)  
    
    return ordem_visita

def buscar_em_profundidade(grade, posicao_inicial):
    altura, largura = grade.shape
    
    pilha = [posicao_inicial]
    visitados = {posicao_inicial}
    ordem_visita = []
    
    while pilha:
        posicao_atual = pilha.pop()  
        ordem_visita.append(posicao_atual)
        
        vizinhos = list(obter_vizinhos(*posicao_atual, altura, largura))
        
        for vizinho in reversed(vizinhos):
            if vizinho not in visitados:
                visitados.add(vizinho)
                pilha.append(vizinho)
    
    return ordem_visita

def mover_um_passo(posicao_atual, posicao_destino):
    (linha_atual, coluna_atual) = posicao_atual
    (linha_destino, coluna_destino) = posicao_destino
    
    if linha_atual < linha_destino:
        return (linha_atual + 1, coluna_atual)  
    if linha_atual > linha_destino:
        return (linha_atual - 1, coluna_atual)  
    if coluna_atual < coluna_destino:
        return (linha_atual, coluna_atual + 1)  
    if coluna_atual > coluna_destino:
        return (linha_atual, coluna_atual - 1) 
    
    return (linha_atual, coluna_atual)  

def animar_aldeoes(grade, total_passos=60, tempo_pausa=0.08):
    altura, largura = grade.shape

    base_aldeia1 = encontrar_posicoes_com_valor(grade, 1)[0]
    base_aldeia2 = encontrar_posicoes_com_valor(grade, 2)[0]

    destinos_bfs = buscar_em_largura(grade, base_aldeia1)  
    destinos_dfs = buscar_em_profundidade(grade, base_aldeia2)  
    
    indice_bfs = 0  
    indice_dfs = 0  

    posicao_aldeao1 = base_aldeia1
    posicao_aldeao2 = base_aldeia2

    plt.ion()  
    cores, normalizacao = configurar_cores_grade()

    figura, eixo = plt.subplots()
    eixo.imshow(grade, cmap=cores, norm=normalizacao)
    
    configurar_grid_visual(eixo, altura, largura)

    marcadores = eixo.scatter([], [], s=120, c=[], edgecolors="white", linewidths=1.0)

    for passo in range(total_passos):
        destino_aldeao1 = destinos_bfs[indice_bfs]
        destino_aldeao2 = destinos_dfs[indice_dfs]

        if posicao_aldeao1 != destino_aldeao1:
            posicao_aldeao1 = mover_um_passo(posicao_aldeao1, destino_aldeao1)
        else:
            indice_bfs = (indice_bfs + 1) % len(destinos_bfs)

        if posicao_aldeao2 != destino_aldeao2:
            posicao_aldeao2 = mover_um_passo(posicao_aldeao2, destino_aldeao2)
        else:
            indice_dfs = (indice_dfs + 1) % len(destinos_dfs)

        posicoes_x = [posicao_aldeao1[1], posicao_aldeao2[1]]  
        posicoes_y = [posicao_aldeao1[0], posicao_aldeao2[0]]  
        cores_aldeoes = ["purple", "orange"]  
        
        marcadores.set_offsets(np.c_[posicoes_x, posicoes_y])
        marcadores.set_color(cores_aldeoes)

        figura.canvas.draw()
        figura.canvas.flush_events()
        plt.pause(tempo_pausa)

    plt.ioff()  
    plt.show()

# teste
alt, larg = 10, 15
g = criar_grade(alt, larg)
g = colocar_aldeias(g, [(2, 2), (7, 12)], ids=[1, 2])
g = colocar_comidas(g, [(1, 4), (3, 10), (8, 2)])

animar_aldeoes(g, total_passos=300, tempo_pausa=0.06)