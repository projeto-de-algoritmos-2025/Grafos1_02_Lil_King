import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from collections import deque
from matplotlib.offsetbox import OffsetImage, AnnotationBbox  
from matplotlib.widgets import Button
import matplotlib.patches as mpatches
import sys

jogo_rodando = False

def criar_grade(alt, larg):
    return np.zeros((alt, larg), dtype=int)

def configurar_cores_grade():
    grass = '#8fcf69'
    cores = mcolors.ListedColormap([grass, "purple", "orange", grass])
    limites = [-0.5, 0.5, 1.5, 2.5, 3.5]
    normalizacao = mcolors.BoundaryNorm(limites, cores.N)
    return cores, normalizacao

def carregar_imagens():
    import matplotlib.image as mpimg
    import os
    
    base_path = os.path.dirname(__file__)
    assets_path = os.path.join(base_path, 'assets')
    
    imagens = {}
    imagens['castelo'] = np.flipud(mpimg.imread(os.path.join(assets_path, 'castelo.png')))
    imagens['cabana'] = np.flipud(mpimg.imread(os.path.join(assets_path, 'cabana.png')))
    imagens['cavaleiro'] = np.flipud(mpimg.imread(os.path.join(assets_path, 'cavaleiro.png')))
    imagens['goblin'] = np.flipud(mpimg.imread(os.path.join(assets_path, 'goblin.png')))
    
    return imagens

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
        return (linha_atual, coluna_destino - 1) 
    
    return (linha_atual, coluna_atual)  

def animar_aldeoes(grade, total_passos=60, tempo_pausa=0.08, figura=None, eixo=None):
    altura, largura = grade.shape

    base_aldeia1 = encontrar_posicoes_com_valor(grade, 1)[0]
    base_aldeia2 = encontrar_posicoes_com_valor(grade, 2)[0]

    destinos_bfs = buscar_em_largura(grade, base_aldeia1)
    destinos_dfs = buscar_em_profundidade(grade, base_aldeia2)

    indice_bfs = 0
    indice_dfs = 0

    posicao_aldeao1 = base_aldeia1
    posicao_aldeao2 = base_aldeia2

    imagens = carregar_imagens()

    created_fig = False
    if figura is None or eixo is None:
        plt.ion()
        cores, normalizacao = configurar_cores_grade()
        figura, eixo = plt.subplots(figsize=(12, 8))
        grade_im = eixo.imshow(grade, cmap=cores, norm=normalizacao,
                   extent=(-0.5, largura-0.5, -0.5, altura-0.5), origin='lower')
        configurar_grid_visual(eixo, altura, largura)
        eixo.set_aspect('equal')
        eixo.set_xlim(-0.5, largura-0.5)
        eixo.set_ylim(-0.5, altura-0.5)
        created_fig = True
    else:
        plt.ion()
        cores, normalizacao = configurar_cores_grade()
        eixo.clear()
        grade_im = eixo.imshow(grade, cmap=cores, norm=normalizacao,
                   extent=(-0.5, largura-0.5, -0.5, altura-0.5), origin='lower')
        configurar_grid_visual(eixo, altura, largura)
        eixo.set_aspect('equal')
        eixo.set_xlim(-0.5, largura-0.5)
        eixo.set_ylim(-0.5, altura-0.5)

    contador_cavaleiro = 0
    contador_goblin = 0
    retornando1 = False
    retornando2 = False
    r1, c1 = base_aldeia1
    r2, c2 = base_aldeia2
    castelo_im = eixo.imshow(imagens['castelo'], extent=(c1-0.5, c1+0.5, r1-0.5, r1+0.5), origin='lower', zorder=2, interpolation='nearest', clip_on=True)
    cabana_im = eixo.imshow(imagens['cabana'], extent=(c2-0.5, c2+0.5, r2-0.5, r2+0.5), origin='lower', zorder=2, interpolation='nearest', clip_on=True)

    cavaleiro_im = eixo.imshow(imagens['cavaleiro'], extent=(posicao_aldeao1[1]-0.5, posicao_aldeao1[1]+0.5, posicao_aldeao1[0]-0.5, posicao_aldeao1[0]+0.5), origin='lower', zorder=3, interpolation='nearest', clip_on=True)
    goblin_im = eixo.imshow(imagens['goblin'], extent=(posicao_aldeao2[1]-0.5, posicao_aldeao2[1]+0.5, posicao_aldeao2[0]-0.5, posicao_aldeao2[0]+0.5), origin='lower', zorder=3, interpolation='nearest', clip_on=True)

    text_cav = eixo.text(0.02, 0.98, f"Cavaleiro: {contador_cavaleiro}", transform=eixo.transAxes, va='top', color='black', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    text_gob = eixo.text(0.98, 0.98, f"Goblin: {contador_goblin}", transform=eixo.transAxes, va='top', ha='right', color='black', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

    food_patches = []
    def desenhar_comidas():
        nonlocal food_patches
        for p in food_patches:
            try:
                p.remove()
            except Exception:
                pass
        food_patches = []
        posicoes_comida = encontrar_posicoes_com_valor(grade, 3)
        for (rr, cc) in posicoes_comida:
            circ = mpatches.Circle((cc, rr), 0.35, facecolor='red', edgecolor='darkred', linewidth=1, zorder=4, clip_on=True)
            eixo.add_patch(circ)
            food_patches.append(circ)

    desenhar_comidas()

    winner = None
    passo = 0
    while True:
        passo += 1
        destino_aldeao1 = base_aldeia1 if retornando1 else destinos_bfs[indice_bfs]
        destino_aldeao2 = base_aldeia2 if retornando2 else destinos_dfs[indice_dfs]

        if posicao_aldeao1 != destino_aldeao1:
            posicao_aldeao1 = mover_um_passo(posicao_aldeao1, destino_aldeao1)
        else:
            if retornando1:
                retornando1 = False
                indice_bfs = 0
            else:
                indice_bfs = (indice_bfs + 1) % len(destinos_bfs)

        if posicao_aldeao2 != destino_aldeao2:
            posicao_aldeao2 = mover_um_passo(posicao_aldeao2, destino_aldeao2)
        else:
            if retornando2:
                retornando2 = False
                indice_dfs = 0
            else:
                indice_dfs = (indice_dfs + 1) % len(destinos_dfs)

        if grade[posicao_aldeao1] == 3 and not retornando1:
            contador_cavaleiro += 1
            grade[posicao_aldeao1] = 0
            vazios = [p for p in encontrar_posicoes_com_valor(grade, 0) if p != posicao_aldeao1 and p != posicao_aldeao2]
            if len(vazios) > 0:
                novo = vazios[np.random.randint(len(vazios))]
                grade[novo] = 3
            try:
                grade_im.set_data(grade)
            except Exception:
                pass
            try:
                desenhar_comidas()
            except Exception:
                pass
            retornando1 = True

        if grade[posicao_aldeao2] == 3 and not retornando2:
            contador_goblin += 1
            grade[posicao_aldeao2] = 0
            vazios = [p for p in encontrar_posicoes_com_valor(grade, 0) if p != posicao_aldeao1 and p != posicao_aldeao2]
            if len(vazios) > 0:
                novo = vazios[np.random.randint(len(vazios))]
                grade[novo] = 3
            try:
                grade_im.set_data(grade)
            except Exception:
                pass
            try:
                desenhar_comidas()
            except Exception:
                pass
            retornando2 = True

        cavaleiro_im.set_extent((posicao_aldeao1[1]-0.5, posicao_aldeao1[1]+0.5, posicao_aldeao1[0]-0.5, posicao_aldeao1[0]+0.5))
        goblin_im.set_extent((posicao_aldeao2[1]-0.5, posicao_aldeao2[1]+0.5, posicao_aldeao2[0]-0.5, posicao_aldeao2[0]+0.5))

        text_cav.set_text(f"Cavaleiro: {contador_cavaleiro}")
        text_gob.set_text(f"Goblin: {contador_goblin}")

        figura.canvas.draw()
        figura.canvas.flush_events()
        plt.pause(tempo_pausa)

        if contador_cavaleiro >= 5:
            winner = 'Cavaleiro'
            break
        if contador_goblin >= 5:
            winner = 'Goblin'
            break

    plt.ioff()
    if winner is not None:
        try:
            jogo_rodando = False
        except Exception:
            pass
        try:
            plt.close(figura)
        except Exception:
            pass

        fig_win, ax_win = plt.subplots(figsize=(6, 5))
        ax_win.axis('off')
        ax_win.set_title('Resultado', fontsize=18)
        ax_win.text(0.5, 0.75, f'Vencedor: {winner}', ha='center', va='center', fontsize=22)
        ax_win.text(0.5, 0.55, f'Cavaleiro: {contador_cavaleiro} — Goblin: {contador_goblin}', ha='center', va='center', fontsize=14)

        try:
            img = imagens['cavaleiro'] if winner == 'Cavaleiro' else imagens['goblin']
            from matplotlib.offsetbox import OffsetImage, AnnotationBbox
            oi = OffsetImage(np.flipud(img), zoom=0.7)
            ab = AnnotationBbox(oi, (0.5, 0.3), frameon=False, xycoords='axes fraction')
            ax_win.add_artist(ab)
        except Exception:
            pass

        plt.show()
    else:
        if created_fig:
            plt.show()
            try:
                jogo_rodando = False
            except Exception:
                pass

def iniciar_jogo_na_figura(figura, eixo, grade, total_passos=300, tempo_pausa=0.06):
    try:
        eixo.clear()
        figura.suptitle("Lil King - Jogo", fontsize=20)
        animar_aldeoes(grade, total_passos=total_passos, tempo_pausa=tempo_pausa, figura=figura, eixo=eixo)
    except Exception:
        animar_aldeoes(grade, total_passos=total_passos, tempo_pausa=tempo_pausa)

def mostrar_tela_inicial():
    imagens = carregar_imagens()

    figura, eixo = plt.subplots(figsize=(8, 6))
    eixo.axis('off')
    figura.suptitle("Lil King", fontsize=28, y=0.95)

    eixo.text(0.5, 0.62, "Bem-vindo(a) ao Lil King\nUma simulação com grafos", ha='center', va='center', transform=eixo.transAxes, fontsize=16)

    ax_jogar = plt.axes([0.34, 0.12, 0.12, 0.08])
    ax_sair = plt.axes([0.54, 0.12, 0.12, 0.08])

    botao_jogar = Button(ax_jogar, 'Jogar')
    botao_sair = Button(ax_sair, 'Sair')

    figura._jogar = False

    def on_jogar(event):
        global jogo_rodando
        if jogo_rodando:
            return
        figura._jogar = True
        try:
            ax_jogar.set_visible(False)
            ax_sair.set_visible(False)
        except Exception:
            pass
        figura.canvas.draw_idle()

        alt, larg = 10, 15
        g = criar_grade(alt, larg)
        g = colocar_aldeias(g, [(2, 2), (7, 12)], ids=[1, 2])
        g = colocar_comidas(g, [(1, 4), (3, 10), (8, 2)])
        timer = figura.canvas.new_timer(interval=100)
        try:
            timer.add_callback(iniciar_jogo_na_figura, figura, eixo, g)
            timer.single_shot = True
        except Exception:
            timer = figura.canvas.new_timer(interval=100, callbacks=[(iniciar_jogo_na_figura, [figura, eixo, g], {})])
        jogo_rodando = True
        timer.start()

    def on_sair(event):
        plt.close('all')
        sys.exit(0)

    botao_jogar.on_clicked(on_jogar)
    botao_sair.on_clicked(on_sair)

    plt.show()

    return getattr(figura, '_jogar', False)

if __name__ == "__main__":
    mostrar_tela_inicial()