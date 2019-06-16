## ex 7.1 -- deck of cards
from enum import Enum
from itertools import it
import random
class Deck:
    SUITS = ['Spades', 'Clubs', 'Hearts', 'Diamonds']
    RANKS = ['A'] + list(2, range(11)) + ['J', 'Q', 'K']

    def __init__(self):
        self.cards = [Card(r, s) for r in self.SUITS for s in self.RANKS]

    def shuffle(self):
        random.shuffle(self.cards)

    def draw(self):
        return self.cards.pop()


class Card:
    def __init__(self, rank, suit):
        self.suit = suit
        self.rank = rank

    def value(self):
        pass

class Hand:
    def __init__(self):
        self.cards = []

    def draw(self, deck):
        self.cards.append(deck.draw())

    def value(self):
        pass




## ex 7.6 -- jigsaw puzzle NxN

class Board:
    def __init__(self, N):
        self.N = N
        self.left = N * N
        self.grid = [[0 for j in range(N)] for j in range(N)]

    def _isComplete(self):
        return self.left == 0

    def contains_piece(self, piece):
        r = piece.row
        c = piece.column
        return self.grid[r][c]

    def place_piece(self, piece, connecting):
        if not self.contains_piece(piece) and self.contains_piece(connecting) and piece.fits_with(connecting):
            self.left -= 1
            self.grid[piece.row][piece.column] = 1
            return True
        else:
            return False

class Piece:
    def __init__(self, r, c, N):
        self.row = r
        self.column = c
        self.metadata = None
        self.N = N

    def fits_with(self, other_piece):
        other_row = other_piece.row
        other_column = other_piece.column
        vertical = abs(other_row - self.row) == 1 and other_column == self.column
        horizontal = abs(other_column - self.column) == 1 and other_row == self.row
        return vertical or horizontal


class JigsawGame:
    def __init__(self, N):
        self.board = Board(N)
        self.pieces = [Piece(i, j, N) for i in range(N) for j in range(N)]
        self.run()

    def run(self):
        while not self.board.isComplete:
            p1, connecting = input()
