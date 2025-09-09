# Lil King — Simulação com Grafos

Uma simulação em Python que usa uma grade para modelar aldeias, unidades (Cavaleiro e Goblin) e recursos (comidas). O objetivo é demonstrar conceitos de grafos e algoritmos de busca (BFS/DFS) aplicados a movimentação e busca de recursos em um ambiente discreto.

Principais características

- Grade 2D com células representando terreno, aldeias e comidas.
- Duas unidades (Cavaleiro e Goblin) que percorrem a grade com rotinas baseadas em BFS e DFS.
- Visualização com imagens (pasta `assets/`) e animação usando Matplotlib.
- Controles básicos para pausar/reiniciar a simulação e definir parâmetros iniciais.

Tecnologias e dependências

- Python 3.10+
- numpy
- matplotlib
- networkx (opcional para extensões de grafo)
- pytest (para testes automatizados)

Link da apresentação do projeto: https://youtu.be/1RalvFeizqs?si=JNjUlmERLucIbW-w

As dependências estão listadas em `requirements.txt`.

Como executar

1. Criar e ativar um ambiente virtual (recomendado):

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Rodar a simulação (modo interativo):

```bash
python main.py
```

3. Executar diretamente com parâmetros (ex.: iniciar automaticamente e configurar grade/meta):

```bash
python main.py --auto --altura 10 --largura 15 --meta 5 --comidas 3 --tempo 0.02
```

Opções úteis (veja `main.py` para todas as flags):

- `--altura` / `--largura`: tamanho da grade
- `--meta`: número de comidas necessárias para vencer
- `--comidas`: número inicial de comidas
- `--tempo`: pausa entre frames (segundos)
- `--auto`: pular a tela inicial e iniciar imediatamente

Executando os testes

```bash
pytest
```

Estrutura do projeto

- `main.py` — código principal da simulação
- `assets/` — imagens usadas na visualização (castelo, cabana, cavaleiro, goblin)
- `tests/` — testes automatizados (ex.: `tests/test_sim.py`)

Observações

- Certifique-se de que a pasta `assets/` contém as imagens esperadas; o programa tenta carregar imagens e usa um placeholder simples caso não as encontre.
- Em sistemas Linux, se for necessário instalar pacotes do sistema para renderização, verifique a sua distribuição (normalmente nada extra é necessário além das dependências Python listadas).

Autores

<table>
  <tr>
    <td align="center"><a href="https://github.com/MM4k"><img style="border-radius: 60%;" src="https://github.com/MM4k.png" width="200px;" alt=""/><br /><sub><b>Marcelo Makoto</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/EnzoEmir"><img style="border-radius: 60%;" src="https://github.com/EnzoEmir.png" width="200px;" alt=""/><br /><sub><b>Enzo Emir</b></sub></a><br /></td>
  </tr>
</table>