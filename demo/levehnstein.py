import re
import os
import spacy
import numpy as np
cimport numpy as np
cimport cython
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
forma_lema_dt = np.dtype([('forma', np.unicode_, 35), ('lema', np.unicode_, 35)])
DEBUG_DEV = False
cdef class Recuperador():
    cdef:
        dict __dict__
        int i
        int j
        int x
        int y
        int size_x
        int size_y
        int w_hash
        str c
        str word
        str ruta_diccionario
        np.ndarray grupos_acentuados
        np.ndarray super_ultra_hasher
        np.ndarray forma_lema_pares
        dict tabla_formas
    def __init__(self, ruta_diccionario:str = BASE_DIR+'/recuperar/datos/dicc.src'):
        self.super_ultra_hasher = np.array([ #filtra, brilla y da esplendor
            'a','b','c','d','e','f','g',
            'h','i','j','k','l', 'm','n','ñ',
            'o','p','q','r','s','t',
            'u','v','w','x','y','z','!',
            '¡', '¿','?','á','é','í','ó',
            'ú', 'ç', 'ü'])
        self.vocales = 'aeiou'
        self.consonantes = 'bcdfghjklmnñpqrstvwxyz'
        self.grupos_acentuados = np.array(['ci?n', 'metr?a', 'az?n', 'er?a'])
        self.ruta_diccionario = ruta_diccionario
        self.regex_words = re.compile(r'[A-Za-záéíóúçñ]*')
        self.filtrar_palabra = lambda churro: re.findall(self.regex_words, churro)
        self.tabla_formas = {}
        self.generar_diccionario()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def hasher(self,word):
        word = word[:6]
        try:
            for c in word:
                indice = np.where(self.super_ultra_hasher == c)
            w_hash =  sum([np.where(self.super_ultra_hasher == c)[0] for c in word]) + len(word)
            return w_hash
        except:
            return None

    @staticmethod
    def levenshtein(seq1, seq2):
        size_x = len(seq1) + 1
        size_y = len(seq2) + 1
        matrix = np.zeros ((size_x, size_y))
        for x in xrange(size_x):
            matrix [x, 0] = x
        for y in xrange(size_y):
            matrix [0, y] = y

        for x in xrange(1, size_x):
            for y in xrange(1, size_y):
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

    def insertar_db(self, cursor: object, tabla: 'comentarios_levenstein', values: list = ['texto']):
        args_db = [ f"`{arg}`" for arg in args_db]
        values = [ f"`{val}`" for val in values]
        query = f"INSERT INTO {tabla} ({','.join(args_db)}) VALUES ()
        INSERT IGNORE INTO `{tabla}` (`encuestaId`, `comentarioId`, `categoriaId`, `valor`) VALUES (%s, %s, %s, %s);"
    def generar_diccionario(self):
        '''
            tomamos las dos primeras "columnas" del diccionario:
            forma - lema - xxx

            hasheamos las  formas para acelerar el lookup
        '''
        with open(self.ruta_diccionario, 'r', encoding='utf-8') as dic:
            lineas = dic.readlines()
            forma_lema_pares = np.fromiter(lineas, dtype=forma_lema_dt, count=len(lineas))
            for i in xrange(len(lineas)):
                linea = [c for c in self.filtrar_palabra(lineas[i]) if c != '']
                forma = linea[0]
                lema = linea[1]
                if len(linea) >= 3:
                    if linea[2][0] != 'A':
                        forma_lema_pares[i]['forma'] = forma.lower()
                        forma_lema_pares[i]['lema'] = lema.lower()
            print('-----------------------')
        for f in xrange(len(forma_lema_pares['forma'])):
            f = forma_lema_pares['forma'][f]
            if any(c in f for c in 'áéíóúñçàèìòùâêîôû'):
                if f[0] not in  self.tabla_formas:
                    self.tabla_formas[f[0]] = [f]
                else:
                    self.tabla_formas[f[0]].append(f)

    def por_caracter(self,texto):
        texto = texto.split(' ')

        for i in range(len(texto)):
            try:
                palabra = texto[i]
                interrogatzia = palabra.index('?')
                if interrogatzia != -1:
                    if palabra[0] != '?':
                        for w in self.tabla_formas[palabra[0]]:
                            distancia = self.levenshtein(palabra, w)
                            if distancia[0] == 1.0:
                                for row in distancia[1]:
                                    target = row[0+int(row[0])]
                                    if target != 0. and palabra[int(row[0])] not in self.consonantes:
                                        if DEBUG_DEV:
                                            print(distancia)
                                            print(palabra, w)
                                            print(f"target: {target} :: [0] {row[0]}")
                                            print(f"reemplazar :: {row}")
                                        texto[i] = w
                    elif palabra[0] == '?':
                        if palabra[1] not in 'rl':
                            for v in 'àèìòù':
                                palabra[0] = v
                                for word in self.tabla_formas[v]:
                                    if palabra == word:
                                        palabra = word
                                        continue
                    '''
                        if palabra[:-1] == '?':
                        
                        if palabra[:-2] not in self.vocales:

                        if 
                        '''
            except:
                next
        return ' '.join(texto)