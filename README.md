# Competição de Agentes Inteligentes em um Labirinto

Projeto desenvolvido para o **Estudo Dirigido** da disciplina de **Inteligência Artificial** do curso de **Bacharelado em Ciência da Computação** da **Universidade Federal do Agreste de Pernambuco (UFAPE)**.

O projeto foi desenvolvido no **Contexto I (Ambiente Próprio)**, conforme proposto na atividade da disciplina.

---

## Informações

- **Disciplina:** Inteligência Artificial
- **Atividade:** Estudo Dirigido
- **Período:** 2026.1
- **Instituição:** Universidade Federal do Agreste de Pernambuco (UFAPE)

### Docente

- [Luis Filipe Alves Pereira](https://github.com/luisfilipeap)

### Discentes

- [Gison Vilaça](https://github.com/gison-vilaca)
- [Vinicius Leite](https://github.com/ViniciusLeiteCosta)

---

# Sobre o Projeto

O objetivo deste projeto é comparar diferentes paradigmas de Inteligência Artificial em um mesmo ambiente.

Foi desenvolvido um **labirinto** onde três agentes inteligentes competem entre si para alcançar o objetivo localizado no centro do mapa.

Cada agente utiliza uma abordagem distinta:

- **Busca Heurística (A\*)**
- **Aprendizado por Reforço (Q-Learning)**
- **Algoritmo Genético**

Os agentes recebem posições iniciais aleatórias dentre um conjunto de pontos de spawn cuidadosamente selecionados para manter a mesma distância mínima até o objetivo. Dessa forma, a competição ocorre em condições equivalentes para todos os participantes.

Ao final de cada rodada é atualizado um placar geral, permitindo comparar o desempenho de cada técnica ao longo da competição.

---

# Funcionalidades

O programa realiza automaticamente:

- geração procedural de um labirinto 17×17;
- criação de caminhos alternativos através da abertura de ciclos;
- seleção de pontos de spawn matematicamente justos;
- competição simultânea entre três agentes inteligentes;
- placar cumulativo entre rodadas;
- interface gráfica utilizando Pygame;
- reinício completo da competição;
- avanço manual para a próxima rodada.

---

# Paradigmas Implementados

## Busca Heurística (A*)

O agente utiliza o algoritmo **A\*** com distância de Manhattan como heurística para encontrar o caminho até o objetivo.

---

## Aprendizado por Reforço (Q-Learning)

O agente realiza um treinamento prévio (offline), aprendendo uma política de navegação através de recompensas e penalidades.

---

## Algoritmo Genético

O agente utiliza uma população de sequências de movimentos que evolui através de:

- seleção por torneio;
- cruzamento;
- mutação;
- elitismo.

---

# Funcionamento do Ambiente

O ambiente é um labirinto gerado automaticamente utilizando o algoritmo **Recursive Backtracker**.

Após sua geração, parte das paredes é removida para criar ciclos e caminhos alternativos, tornando o ambiente mais interessante para comparação entre os agentes.

Para garantir justiça na competição:

- todas as células da borda têm sua distância até o objetivo calculada por **Busca em Largura (BFS)**;
- apenas células com exatamente a mesma distância são consideradas candidatas;
- os agentes são distribuídos aleatoriamente entre esses pontos.

Assim, nenhum agente inicia mais próximo do objetivo que outro.

---

# Sistema de Pontuação

Cada rodada concede:

| Colocação | Pontos |
|-----------|--------|
| 🥇 1º | 3 |
| 🥈 2º | 2 |
| 🥉 3º | 1 |

Caso algum agente não alcance o objetivo dentro do tempo disponível, o desempate ocorre considerando a distância restante até o centro do labirinto.

---

# Tecnologias Utilizadas

- Python 3.9+
- Pygame

---

# Estrutura do Projeto

```
.
├── README.md
├── requirements.txt
├── main.py
├── game.py
├── maze.py
├── agents.py
└── __pycache__/
```

| Arquivo | Descrição |
|---------|-----------|
| `main.py` | Ponto de entrada da aplicação |
| `game.py` | Interface gráfica, placar, botões e lógica da competição |
| `maze.py` | Geração do labirinto, BFS e seleção dos spawns |
| `agents.py` | Implementação dos agentes A*, Q-Learning e Algoritmo Genético |

---

# Instalação

Clone o repositório:

```bash
git clone https://github.com/gison-vilaca/ia-agentes-inteligentes.git
```

Entre na pasta do projeto:

```bash
cd ia-agentes-inteligentes
```

Crie um ambiente virtual (opcional, mas recomendado):

```bash
python -m venv .venv
```

Ative o ambiente virtual.

Windows (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

---

# Execução

Execute:

```bash
python main.py
```

Uma janela será aberta contendo:

- o labirinto;
- os três agentes;
- o placar;
- os botões de controle da competição.

---

# Parâmetros Configuráveis

Os principais parâmetros podem ser alterados diretamente no código:

**game.py**

- tamanho do labirinto;
- velocidade da simulação;
- número de rodadas;
- tamanho das células;
- quantidade de conexões extras.

**agents.py**

- taxa de aprendizado;
- fator de desconto;
- taxa de exploração;
- tamanho da população;
- taxa de mutação;
- quantidade de gerações.

---

# Aspectos de Inteligência Artificial

Durante a implementação foram aplicados conceitos como:

- representação de estados;
- espaço de estados;
- ações;
- objetivo;
- busca heurística;
- função heurística;
- aprendizado por reforço;
- recompensas;
- algoritmo genético;
- função de aptidão;
- seleção;
- cruzamento;
- mutação.

---

# Licença

Este projeto foi desenvolvido exclusivamente para fins acadêmicos como atividade da disciplina de Inteligência Artificial da UFAPE.