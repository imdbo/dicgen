import re
import os
import numpy as np
import time 

class Levenshtein():
    def __init__(self):
        self.super_ultra_hasher = { #filtra, brilla y da esplendor
            'a': 0,
            'b': 1,
            'c': 2,
            'd': 3,
            'e': 4,
            'f': 5,
            'g': 6,
            'h': 7,
            'i': 8,
            'j': 9,
            'k': 10,
            'l': 11,
            'm': 12,
            'n': 13,
            'ñ': 14,
            'o': 15,
            'p': 16,
            'q': 17,
            'r': 18,
            's': 19,
            't': 20,
            'u': 22,
            'v': 23,
            'w': 24,
            'x': 25,
            'y': 26,
            'z': 27,
            '!': 28,
            '¡': 29, 
            '¿': 30,
            '?': 31,
            'á': 32,
            'é': 33,
            'í': 34,
            'ó': 35,
            'ú': 36,
            'ç': 37,
            'ü': 38}
        self.total_characters = 38

    def levenshtein(self, seq1, seq2):
        size_x = len(seq1) + 1
        size_y = len(seq2) + 1
        matrix = np.zeros ((size_x, size_y))
        for x in range(size_x):
            matrix [x, 0] = x
        for y in range(size_y):
            matrix [0, y] = y

        for x in range(1, size_x):
            for y in range(1, size_y):
                if seq1[x-1] == seq2[y-1]:
                    matrix [x,y] = min(
                        matrix[x-1, y] + 1,
                        matrix[x-1, y-1],
                        matrix[x, y-1] + 1
                    )
                else:
                    matrix [x,y] = min(
                        matrix[x-1,y] + 1,
                        matrix[x-1,y-1] + 1,
                        matrix[x,y-1] + 1
                    )
        return (matrix[size_x - 1, size_y - 1], matrix)
    def add_c(self,c):
        self.total_characters += 1
        self.super_ultra_hasher[c] = self.total_characters

    def find_distance(self, word_1:str, lemma_list:list):
        results = []
        for word_2 in lemma_list:
            for c in word_1+word_2:
                if c not in self.super_ultra_hasher:
                    self.total_characters += 1
                    self.super_ultra_hasher[c] = self.total_characters
            distance = self.levenshtein([self.super_ultra_hasher[c] for c in word_1], [self.super_ultra_hasher[c] for c in word_2])
            if distance[0] <= 1.0:
                results.append(word_2)
        return results