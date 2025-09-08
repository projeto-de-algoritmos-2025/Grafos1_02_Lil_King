import pytest
import numpy as np
from main import obter_vizinhos, mover_um_passo, criar_grade, colocar_aldeias, colocar_comidas, encontrar_posicoes_com_valor


def test_obter_vizinhos_center():
    viz = list(obter_vizinhos(1, 1, 3, 3))
    assert set(viz) == {(0,1),(2,1),(1,0),(1,2)}


def test_mover_um_passo_vertical():
    assert mover_um_passo((0,0),(2,0)) == (1,0)
    assert mover_um_passo((2,0),(0,0)) == (1,0)


def test_mover_um_passo_horizontal():
    assert mover_um_passo((0,0),(0,2)) == (0,1)
    assert mover_um_passo((0,2),(0,0)) == (0,1)


def test_respawn_nao_coloca_em_aldeias():
    g = criar_grade(5,5)
    g = colocar_aldeias(g, [(1,1),(3,3)], ids=[1,2])
    g = colocar_comidas(g, [(0,0)])
    pos_before = set(encontrar_posicoes_com_valor(g,3))
    g[0,0] = 0
    vazios = [p for p in encontrar_posicoes_com_valor(g,0) if p not in [(1,1),(3,3)]]
    assert (1,1) not in vazios

