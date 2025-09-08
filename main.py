import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from collections import deque
from matplotlib.offsetbox import OffsetImage, AnnotationBbox  
from matplotlib.widgets import Button, TextBox
import matplotlib.patches as mpatches
import sys
import argparse
import os

jogo_rodando = False
CONFIG = {}

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

    base_path = os.path.dirname(__file__)
    assets_path = os.path.join(base_path, 'assets')

    def try_load(name, filename):
        p = os.path.join(assets_path, filename)
        try:
            if os.path.exists(p):
                return np.flipud(mpimg.imread(p))
        except Exception:
            pass
        return np.ones((8, 8, 3), dtype=float) * 0.5

    imagens = {}
    imagens['castelo'] = try_load('castelo', 'castelo.png')
    imagens['cabana'] = try_load('cabana', 'cabana.png')
    imagens['cavaleiro'] = try_load('cavaleiro', 'cavaleiro.png')
    imagens['goblin'] = try_load('goblin', 'goblin.png')

    return imagens

def configurar_grid_visual(eixo, altura, largura):
    eixo.set_xticks(np.arange(-0.5, largura, 1), minor=True)
    eixo.set_yticks(np.arange(-0.5, altura, 1), minor=True)
    eixo.grid(which="minor", color="black", linestyle="-", linewidth=0.5)
    for c in range(largura + 1):
        eixo.axvline(c - 0.5, color='black', linewidth=0.5, zorder=0)
    for r in range(altura + 1):
        eixo.axhline(r - 0.5, color='black', linewidth=0.5, zorder=0)
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

def gerar_posicoes_comida(grade, n, exclude=None):
    """Gera até n posições vazias (valor 0) na grade, excluindo coordenadas em `exclude`."""
    if exclude is None:
        exclude = []
    altura, largura = grade.shape
    vazios = [tuple(x) for x in np.argwhere(grade == 0)]
    vazios = [p for p in vazios if p not in exclude]
    if n >= len(vazios):
        return vazios
    idx = np.random.choice(len(vazios), size=n, replace=False)
    return [vazios[i] for i in idx]

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


def encontrar_caminho_bfs(grade, inicio, destino):
    """Retorna lista de coordenadas do caminho mais curto (BFS) entre inicio e destino inclusive."""
    altura, largura = grade.shape
    from collections import deque
    fila = deque([inicio])
    veio_de = {inicio: None}

    while fila:
        atual = fila.popleft()
        if atual == destino:
            break
        for v in obter_vizinhos(*atual, altura, largura):
            if v not in veio_de:
                veio_de[v] = atual
                fila.append(v)

    if destino not in veio_de:
        return []

    caminho = []
    node = destino
    while node is not None:
        caminho.append(node)
        node = veio_de[node]
    caminho.reverse()
    return caminho

def animar_aldeoes(grade, total_passos=60, tempo_pausa=0.02, figura=None, eixo=None, meta=5):
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

    pausado = False
    reiniciar = False

    def toggle_pause(event):
        nonlocal pausado
        pausado = not pausado
        btn_pause.label.set_text('Resume' if pausado else 'Pause')

    def do_restart(event):
        nonlocal reiniciar
        reiniciar = True

    ax_pause = figura.add_axes([0.02, 0.02, 0.08, 0.06])
    ax_restart = figura.add_axes([0.12, 0.02, 0.08, 0.06])
    btn_pause = Button(ax_pause, 'Pause')
    btn_restart = Button(ax_restart, 'Restart')
    btn_pause.on_clicked(toggle_pause)
    btn_restart.on_clicked(do_restart)

    winner = None
    passo = 0

    while True:
        passo += 1

        while pausado:
            figura.canvas.draw()
            figura.canvas.flush_events()
            plt.pause(0.05)
            if reiniciar:
                break

        if reiniciar:
            altura, largura = grade.shape
            bases = [p for p in encontrar_posicoes_com_valor(grade, 1) + encontrar_posicoes_com_valor(grade, 2)]
            if len(bases) >= 2:
                base_aldeia1 = bases[0]
                base_aldeia2 = bases[1]
            else:
                base_aldeia1 = (2,2)
                base_aldeia2 = (7,12) if largura > 12 and altura > 7 else (altura-2, largura-2)
            destinos_bfs = buscar_em_largura(grade, base_aldeia1)
            destinos_dfs = buscar_em_profundidade(grade, base_aldeia2)
            indice_bfs = 0
            indice_dfs = 0
            posicao_aldeao1 = base_aldeia1
            posicao_aldeao2 = base_aldeia2
            contador_cavaleiro = 0
            contador_goblin = 0
            retornando1 = False
            retornando2 = False
            desenhar_comidas()
            reiniciar = False
            continue
        destino_aldeao1 = base_aldeia1 if retornando1 else destinos_bfs[indice_bfs]
        destino_aldeao2 = base_aldeia2 if retornando2 else destinos_dfs[indice_dfs]

        if posicao_aldeao1 != destino_aldeao1:
            posicao_aldeao1 = mover_um_passo(posicao_aldeao1, destino_aldeao1)
            cavaleiro_im.set_extent((posicao_aldeao1[1]-0.5, posicao_aldeao1[1]+0.5, posicao_aldeao1[0]-0.5, posicao_aldeao1[0]+0.5))
        else:
            if retornando1:
                retornando1 = False
                indice_bfs = 0
            else:
                indice_bfs = (indice_bfs + 1) % len(destinos_bfs)

        if posicao_aldeao2 != destino_aldeao2:
            posicao_aldeao2 = mover_um_passo(posicao_aldeao2, destino_aldeao2)
            goblin_im.set_extent((posicao_aldeao2[1]-0.5, posicao_aldeao2[1]+0.5, posicao_aldeao2[0]-0.5, posicao_aldeao2[0]+0.5))
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

        if contador_cavaleiro >= meta:
            winner = 'Cavaleiro'
            break
        if contador_goblin >= meta:
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

def iniciar_jogo_na_figura(figura, eixo, grade, total_passos=300, tempo_pausa=0.02, meta=5):
    try:
        eixo.clear()
        figura.suptitle("Lil King - Jogo", fontsize=20)
        animar_aldeoes(grade, total_passos=total_passos, tempo_pausa=tempo_pausa, figura=figura, eixo=eixo, meta=meta)
    except Exception:
        animar_aldeoes(grade, total_passos=total_passos, tempo_pausa=tempo_pausa, meta=meta)

def mostrar_tela_inicial():
    imagens = carregar_imagens()

    figura, eixo = plt.subplots(figsize=(8, 6))
    eixo.axis('off')
    sup = figura.suptitle("Lil King", fontsize=28, y=0.95)

    welcome_txt = eixo.text(0.5, 0.62, "Bem-vindo(a) ao Lil King\nUma simulação com grafos", ha='center', va='center', transform=eixo.transAxes, fontsize=16)

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
            sup.set_visible(False)
            welcome_txt.set_visible(False)
        except Exception:
            pass
        figura.canvas.draw_idle()

        cfg_axs = {}
        start_y = 0.60
        gap = 0.08
        cfg_axs['alt'] = figura.add_axes([0.35, start_y, 0.3, 0.05])
        cfg_axs['larg'] = figura.add_axes([0.35, start_y-gap, 0.3, 0.05])
        cfg_axs['meta'] = figura.add_axes([0.35, start_y-2*gap, 0.3, 0.05])
        cfg_axs['comidas'] = figura.add_axes([0.35, start_y-3*gap, 0.3, 0.05])

        tb_alt = TextBox(cfg_axs['alt'], 'Altura', initial=str(CONFIG.get('altura', 10)))
        tb_larg = TextBox(cfg_axs['larg'], 'Largura', initial=str(CONFIG.get('largura', 15)))
        tb_meta = TextBox(cfg_axs['meta'], 'Meta', initial=str(CONFIG.get('meta', 5)))
        tb_comidas = TextBox(cfg_axs['comidas'], 'Comidas', initial=str(CONFIG.get('comidas', 3)))

        ax_iniciar = figura.add_axes([0.35, start_y-5*gap-0.02, 0.12, 0.06])
        ax_cancel = figura.add_axes([0.53, start_y-5*gap-0.02, 0.12, 0.06])
        btn_iniciar = Button(ax_iniciar, 'Iniciar')
        btn_cancel = Button(ax_cancel, 'Cancelar')

        figura._cfg_widgets = [cfg_axs, tb_alt, tb_larg, tb_meta, tb_comidas, ax_iniciar, ax_cancel, btn_iniciar, btn_cancel]

        def limpar_cfg(restore=True):
            try:
                for a in cfg_axs.values():
                    a.remove()
            except Exception:
                pass
            try:
                ax_iniciar.remove(); ax_cancel.remove()
            except Exception:
                pass
            if restore:
                try:
                    ax_jogar.set_visible(True); ax_sair.set_visible(True)
                except Exception:
                    pass
                try:
                    sup.set_visible(True)
                    welcome_txt.set_visible(True)
                except Exception:
                    pass
            try:
                del figura._cfg_widgets
            except Exception:
                pass
            figura.canvas.draw_idle()

        def on_iniciar(event):
            global jogo_rodando
            try:
                alt = int(tb_alt.text)
                larg = int(tb_larg.text)
                meta = int(tb_meta.text)
                n_comidas = int(tb_comidas.text)
            except Exception:
                return
            tempo = float(CONFIG.get('tempo', 0.02))
            CONFIG.update(dict(altura=alt, largura=larg, meta=meta, comidas=n_comidas))
            g = criar_grade(alt, larg)
            g = colocar_aldeias(g, [(2, 2), (max(1, alt-3), max(1, larg-3))], ids=[1, 2])
            aldeias = encontrar_posicoes_com_valor(g, 1) + encontrar_posicoes_com_valor(g, 2)
            pos_comidas = gerar_posicoes_comida(g, CONFIG.get('comidas', 3), exclude=aldeias)
            g = colocar_comidas(g, pos_comidas)
            jogo_rodando = True
            limpar_cfg(restore=False)
            try:
                iniciar_jogo_na_figura(figura, eixo, g, total_passos=300, tempo_pausa=tempo, meta=meta)
            except Exception:
                timer = figura.canvas.new_timer(interval=100)
                try:
                    timer.add_callback(iniciar_jogo_na_figura, figura, eixo, g, 300, tempo, meta)
                    timer.single_shot = True
                    timer.start()
                except Exception:
                    pass

        def on_cancel(event):
            limpar_cfg()

        btn_iniciar.on_clicked(on_iniciar)
        btn_cancel.on_clicked(on_cancel)

    def on_sair(event):
        plt.close('all')
        sys.exit(0)

    botao_jogar.on_clicked(on_jogar)
    botao_sair.on_clicked(on_sair)

    plt.show()

    return getattr(figura, '_jogar', False)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--altura', type=int, default=10, help='altura da grade')
    p.add_argument('--largura', type=int, default=15, help='largura da grade')
    p.add_argument('--meta', type=int, default=5, help='numero de comidas para vencer')
    p.add_argument('--tempo', type=float, default=0.02, help='tempo de pausa entre frames (segundos)')
    p.add_argument('--seed', type=int, default=None, help='seed aleatorio')
    p.add_argument('--comidas', type=int, default=3, help='numero inicial de comidas')
    p.add_argument('--auto', action='store_true', help='iniciar jogo automaticamente (pular tela inicial)')
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)
    CONFIG.update(dict(altura=args.altura, largura=args.largura, meta=args.meta, tempo=args.tempo, comidas=args.comidas))

    if args.auto:
        alt, larg = args.altura, args.largura
        g = criar_grade(alt, larg)
        g = colocar_aldeias(g, [(2, 2), (max(1, alt-3), max(1, larg-3))], ids=[1, 2])
        aldeias = encontrar_posicoes_com_valor(g, 1) + encontrar_posicoes_com_valor(g, 2)
        pos_comidas = gerar_posicoes_comida(g, args.comidas, exclude=aldeias)
        g = colocar_comidas(g, pos_comidas)
        fig, ax = plt.subplots(figsize=(12, 8))
        iniciar_jogo_na_figura(fig, ax, g, total_passos=300, tempo_pausa=args.tempo)
    else:
        mostrar_tela_inicial()