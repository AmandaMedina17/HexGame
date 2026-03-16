from player import Player
from board import HexBoard
from collections import deque
import numpy as np
import random
import math
import time

class SmartPlayer(Player):
    # la profundidad maxima de busqueda (niveles en el arbol)
    max_depth = 3
    # tiempo maximo permitido por jugada en segundos
    time_limit = 5.0

    def play(self, board: HexBoard) -> tuple:
        """
        Metodo principal que la IA usa para decidir su siguiente movimiento.
        Recibe el tablero actual y devuelve una tupla (fila, columna).
        """
        # si el jugador no tiene ninguna ficha, es el primer movimiento
        if not self.get_player_positions(board, self.player_id):
            # jugar en el centro o una celda adyacente
            return self.get_center(board)

        # 1. verificar si podemos ganar en este turno
        win_move = self.find_winning_move(board, self.player_id)
        if win_move:
            return win_move

        # 2. verificar si el oponente puede ganar en su turno y bloquearlo
        block_move = self.find_winning_move(board, 3 - self.player_id)
        if block_move:
            return block_move

        # 3. si no hay jugadas inmediatas, usar busqueda avanzada
        return self.iterative_deepening(board)


    # funciones auxiliares para obtener informacion del tablero

    def get_player_positions(self, board: HexBoard, player_id: int) -> set:
        """
        recorre todo el tablero y devuelve un conjunto con las coordenadas
        de todas las fichas pertenecientes al jugador indicado.
        """
        positions = set()
        for i in range(board.size):
            for j in range(board.size):
                if board.board[i][j] == player_id:
                    positions.add((i, j))
        return positions

    def get_possible_moves(self, board: HexBoard) -> list:
        """devuelve una lista con todas las casillas vacias del tablero"""
        return [(i, j) for i in range(board.size) for j in range(board.size) if board.board[i][j] == 0]

    def get_center(self, board: HexBoard) -> tuple:
        """
        intenta jugar en la celda central; si esta ocupada, elige una
        celda vacia aleatoria entre las 6 adyacentes.
        si todo falla, devuelve la primera casilla vacia disponible.
        """
        size = board.size
        center = (size // 2, size // 2)
        if board.board[center[0]][center[1]] == 0:
            return center
        # direcciones posibles en hex: 6 vecinos
        directions = [(-1,0), (1,0), (0,-1), (0,1), (-1,1), (1,-1)]
        random.shuffle(directions)
        for dr, dc in directions:
            r, c = center[0] + dr, center[1] + dc
            if 0 <= r < size and 0 <= c < size and board.board[r][c] == 0:
                return (r, c)
        # si nada funciona, devolvemos la primera vacia 
        return self.get_possible_moves(board)[0]

    def find_winning_move(self, board: HexBoard, player: int) -> tuple:
        """
        prueba cada movimiento posible para el jugador 'player'
        y devuelve el primero que complete la conexion (victoria).
        si no hay ninguno, retorna None.
        """
        for move in self.get_possible_moves(board):
            new_board = board.clone()
            new_board.place_piece(move[0], move[1], player)
            if new_board.check_connection(player):
                return move
        return None


    # evaluacion heuristica (distancia minima a la conexion)
    def evaluate(self, board: HexBoard) -> float:
        """
        calcula la calidad de la posicion desde el punto de vista del jugador actual.
        usa la diferencia de distancias: (distancia del oponente - distancia propia) * 10.
        valores positivos favorecen a la ia, negativos la perjudican.
        si la ia ya conecto, retorna infinito; si el oponente ya conecto, retorna -infinito.
        """
        my_dist = self.calculate_min_distance_bfs01(board, self.player_id)
        opp_dist = self.calculate_min_distance_bfs01(board, 3 - self.player_id)
        if my_dist == 0:
            return math.inf
        if opp_dist == 0:
            return -math.inf
        return (opp_dist - my_dist) * 10

    def calculate_min_distance_bfs01(self, board: HexBoard, player_id: int) -> float:
        """
        implementa un bfs 0-1 para calcular la distancia minima (en terminos de
        casillas vacias) que necesita el jugador 'player_id' para conectar sus dos lados.

        - las casillas propias tienen coste 0.
        - las casillas vacias tienen coste 1.
        - las casillas del oponente son intransitables.

        devuelve el coste minimo acumulado para llegar desde el lado inicial
        hasta cualquier celda del lado final. si no hay camino, retorna 1000.
        """
        size = board.size
        dist = np.full((size, size), np.inf)   # matriz de distancias inicializada a infinito
        dq = deque()

        # definir los puntos de inicio y los objetivos segun el jugador
        if player_id == 1:  # jugador 1 conecta izquierda (col 0) a derecha (col size-1)
            starts = [(i, 0) for i in range(size) if board.board[i][0] != 3 - player_id]
            target = [(i, size-1) for i in range(size)]
        else:               # jugador 2 conecta arriba (fila 0) a abajo (fila size-1)
            starts = [(0, j) for j in range(size) if board.board[0][j] != 3 - player_id]
            target = [(size-1, j) for j in range(size)]

        # inicializar las celdas de inicio
        for (r, c) in starts:
            cost = 0 if board.board[r][c] == player_id else 1
            dist[r][c] = cost
            if cost == 0:
                dq.appendleft((r, c))   # coste 0 va al frente de la cola
            else:
                dq.append((r, c))        # coste 1 va al final

        # direcciones posibles en el tablero hexagonal
        directions = [(-1,0), (1,0), (0,-1), (0,1), (-1,1), (1,-1)]

        while dq:
            r, c = dq.popleft()
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size and board.board[nr][nc] != 3 - player_id:
                    # coste adicional: 0 si es propia, 1 si es vacia
                    new_cost = dist[r][c] + (0 if board.board[nr][nc] == player_id else 1)
                    if new_cost < dist[nr][nc]:
                        dist[nr][nc] = new_cost
                        if new_cost == 0:
                            dq.appendleft((nr, nc))
                        else:
                            dq.append((nr, nc))

        # buscar la minima distancia entre todas las celdas objetivo
        min_dist = np.inf
        for (r, c) in target:
            if not np.isinf(dist[r][c]):
                min_dist = min(min_dist, dist[r][c])
        # si no se encontro camino, devolver un valor grande (1000)
        return min_dist if not np.isinf(min_dist) else 1000


    # generacion y ordenacion de movimientos candidatos
    def candidate_moves(self, board: HexBoard) -> list:
        """
        devuelve una lista de posibles movimientos (casillas vacias)
        ordenados por prioridad: primero las mas cercanas a las fichas propias
        y al centro. esto mejora la poda en la busqueda.
        """
        empty = set(self.get_possible_moves(board))
        if not empty:
            return []

        # obtener posiciones propias y del oponente
        my_positions = self.get_player_positions(board, self.player_id)
        opp_positions = self.get_player_positions(board, 3 - self.player_id)

        # conjunto de celdas adyacentes a cualquier ficha (propia o del oponente)
        adjacent = set()
        for (r, c) in my_positions | opp_positions:
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1), (-1,1), (1,-1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < board.size and 0 <= nc < board.size and board.board[nr][nc] == 0:
                    adjacent.add((nr, nc))

        # si hay pocas fichas en total, anadir tambien celdas alrededor del centro
        total_pieces = len(my_positions) + len(opp_positions)
        if total_pieces < 5:
            center = board.size // 2
            for r in range(center-1, center+2):
                for c in range(center-1, center+2):
                    if 0 <= r < board.size and 0 <= c < board.size and board.board[r][c] == 0:
                        adjacent.add((r, c))

        # si no se encontro ninguna adyacente, devolver todas las vacias
        if not adjacent:
            return list(empty)

        # funcion de puntuacion para ordenar los movimientos
        def move_score(move):
            r, c = move
            # distancia al centro (manhattan, aunque en hex no es perfecta, pero sirve)
            center_dist = abs(r - board.size//2) + abs(c - board.size//2)
            # distancia a la ficha propia mas cercana
            if my_positions:
                own_dist = min(abs(r - pr) + abs(c - pc) for (pr, pc) in my_positions)
            else:
                own_dist = 0
            # se prioriza la cercania a las propias (por eso se multiplica por 2)
            return own_dist * 2 + center_dist

        # intersectamos las adyacentes con las vacias y ordenamos
        moves = list(adjacent & empty)
        moves.sort(key=move_score)
        return moves


    # busqueda alfa-beta con profundidad iterativa
    def iterative_deepening(self, board: HexBoard) -> tuple:
        """
        realiza busqueda aumentando la profundidad progresivamente hasta
        agotar el tiempo disponible o alcanzar la profundidad maxima.
        devuelve el mejor movimiento encontrado.
        """
        start_time = time.time()
        best_move = None
        for depth in range(1, self.max_depth + 1):
            # si hemos consumido el 90% del tiempo, paramos
            if time.time() - start_time > self.time_limit * 0.9:
                break
            move, score = self.alpha_beta_search(board, depth, start_time)
            if move is not None:
                best_move = move
            # si encontramos una jugada que da la victoria, podemos terminar antes
            if score == math.inf:
                break
        # si no encontramos ningun movimiento (por tiempo), usamos el primer candidato
        return best_move if best_move else self.candidate_moves(board)[0]

    def alpha_beta_search(self, board: HexBoard, depth: int, start_time) -> tuple:
        """
        funcion de llamada para la busqueda alfa-beta.
        retorna (mejor_movimiento, valor_estimado) para la profundidad dada.
        """
        moves = self.candidate_moves(board)
        if not moves:
            return (None, self.evaluate(board))

        best_move = moves[0]
        best_value = -math.inf
        alpha = -math.inf
        beta = math.inf

        for move in moves:
            new_board = board.clone()
            new_board.place_piece(move[0], move[1], self.player_id)
            value = self.min_value(new_board, depth-1, alpha, beta, start_time)
            if value > best_value:
                best_value = value
                best_move = move
            alpha = max(alpha, best_value)
            if best_value >= beta:   # poda
                break
        return (best_move, best_value)

    def max_value(self, board: HexBoard, depth: int, alpha: float, beta: float, start_time) -> float:
        """
        nodo maximizador (turno de la ia).
        """
        if time.time() - start_time > self.time_limit:
            return self.evaluate(board)
        if depth == 0 or board.check_connection(self.player_id):
            return self.evaluate(board)

        moves = self.candidate_moves(board)
        if not moves:
            return self.evaluate(board)

        value = -math.inf
        for move in moves:
            new_board = board.clone()
            new_board.place_piece(move[0], move[1], self.player_id)
            value = max(value, self.min_value(new_board, depth-1, alpha, beta, start_time))
            alpha = max(alpha, value)
            if value >= beta:   # poda
                break
        return value

    def min_value(self, board: HexBoard, depth: int, alpha: float, beta: float, start_time) -> float:
        """
        nodo minimizador (turno del oponente).
        """
        if time.time() - start_time > self.time_limit:
            return self.evaluate(board)
        if depth == 0 or board.check_connection(3 - self.player_id):
            return self.evaluate(board)

        moves = self.candidate_moves(board)
        if not moves:
            return self.evaluate(board)

        value = math.inf
        for move in moves:
            new_board = board.clone()
            new_board.place_piece(move[0], move[1], 3 - self.player_id)
            value = min(value, self.max_value(new_board, depth-1, alpha, beta, start_time))
            beta = min(beta, value)
            if value <= alpha:   # poda
                break
        return value