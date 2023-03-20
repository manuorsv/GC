#!/usr/bin/env python
# coding: utf-8

"""
Práctica 2
"""

import os
import numpy as np
import pandas as pd
import math

#### Vamos al directorio de trabajo####
os.getcwd()
#os.chdir(ubica)
#files = os.listdir(ruta)

with open('GCOM2022_pract2_auxiliar_eng.txt', 'r',encoding="utf8") as file:
      en = file.read()
     
with open('GCOM2022_pract2_auxiliar_esp.txt', 'r',encoding="utf8") as file:
      es = file.read()


#### Contamos cuantas letras hay en cada texto
from collections import Counter
tab_en = Counter(en)
tab_es = Counter(es)

##### Transformamos en formato array de los carácteres (states) y su frecuencia
##### Finalmente realizamos un DataFrame con Pandas y ordenamos con 'sort'
tab_en_states = np.array(list(tab_en))
tab_en_weights = np.array(list(tab_en.values()))
tab_en_probab = tab_en_weights/float(np.sum(tab_en_weights))
distr_en = pd.DataFrame({'states': tab_en_states, 'probab': tab_en_probab})
distr_en = distr_en.sort_values(by='probab', ascending=True)
distr_en.index=np.arange(0,len(tab_en_states))

tab_es_states = np.array(list(tab_es))
tab_es_weights = np.array(list(tab_es.values()))
tab_es_probab = tab_es_weights/float(np.sum(tab_es_weights))
distr_es = pd.DataFrame({'states': tab_es_states, 'probab': tab_es_probab })
distr_es = distr_es.sort_values(by='probab', ascending=True)
distr_es.index=np.arange(0,len(tab_es_states))

##### Para obtener una rama, fusionamos los dos states con menor frecuencia
distr = distr_en
''.join(distr['states'][[0,1]])

### Es decir:
states = np.array(distr['states'])
probab = np.array(distr['probab'])
state_new = np.array([''.join(states[[0,1]])])   #Ojo con: state_new.ndim
probab_new = np.array([np.sum(probab[[0,1]])])   #Ojo con: probab_new.ndim
codigo = np.array([{states[0]: 0, states[1]: 1}])
states =  np.concatenate((states[np.arange(2,len(states))], state_new), axis=0)
probab =  np.concatenate((probab[np.arange(2,len(probab))], probab_new), axis=0)
distr = pd.DataFrame({'states': states, 'probab': probab, })
distr = distr.sort_values(by='probab', ascending=True)
distr.index=np.arange(0,len(states))

#Creamos un diccionario
branch = {'distr':distr, 'codigo':codigo}

## Ahora definimos una función que haga exáctamente lo mismo
def huffman_branch(distr):
    states = np.array(distr['states'])
    probab = np.array(distr['probab'])
    state_new = np.array([''.join(states[[0,1]])])
    probab_new = np.array([np.sum(probab[[0,1]])])
    codigo = np.array([{states[0]: 0, states[1]: 1}])
    states =  np.concatenate((states[np.arange(2,len(states))], state_new), axis=0)
    probab =  np.concatenate((probab[np.arange(2,len(probab))], probab_new), axis=0)
    distr = pd.DataFrame({'states': states, 'probab': probab, })
    distr = distr.sort_values(by='probab', ascending=True)
    distr.index=np.arange(0,len(states))
    branch = {'distr':distr, 'codigo':codigo}
    return(branch) 

def huffman_tree(distr):
    tree = np.array([])
    while len(distr) > 1:
        branch = huffman_branch(distr)
        distr = branch['distr']
        code = np.array([branch['codigo']])
        tree = np.concatenate((tree, code), axis=None)
    return(tree)


def huffman_cod(tree):
    code_dict = {}
    #Recorremos el arbol construyendo la codificacion de cada estado empezando en la raiz
    #Partimos de la raíz para tener una entrada en el diccionario para cada caracter.
    root_left = list(tree[tree.size-1].items())[0][0]
    root_right = list(tree[tree.size-1].items())[1][0]
    for sym in root_left:
        code_dict[sym] = '0'
    for sym in root_right:
        code_dict[sym] = '1'
    
    #Ahora vamos recorriendo el resto de nodos y construyendo la codificación de cada caracter.
    for i in range(tree.size-2,-1,-1):
        left = list(tree[i].items())[0][0]
        right = list(tree[i].items())[1][0]
        
        for sym in left:
            code_dict[sym] += '0'
        for sym in right:
            code_dict[sym] += '1'
            
    return code_dict

#Calcula la longitud media
def long_media(distr, dic):
    long_media = 0
    for i in range(len(distr)):
        w_i = distr['probab'][i]
        c_i = len(dic[distr['states'][i]])
        long_media += w_i*c_i
        
    return long_media

#Calcula la entropía
def entropia(distr):
    entr = 0
    for prob in distr['probab']:
        entr -= prob*math.log(prob,2)
    
    return entr

tree_en = huffman_tree(distr_en)
cod_en = huffman_cod(tree_en)
long_en = long_media(distr_en, cod_en)
entr_en = entropia(distr_en)

#print(cod_en)
print("Longitud media en inglés: " + str(long_en))
print("Entropía en inglés: " + str(entr_en))
print("Se satisface el Primer Teorema de Shannon pues: " + str(entr_en) + ' <= ' + str(long_en) + ' < ' + str(entr_en + 1))

tree_es = huffman_tree(distr_es)
cod_es = huffman_cod(tree_es)
long_es = long_media(distr_es, cod_es)
entr_es = entropia(distr_es)

#print(cod_es)
print("Longitud media en español: " + str(long_es))
print("Entropía en español: " + str(entr_es))
print("Se satisface el Primer Teorema de Shannon pues: " + str(entr_es) + ' <= ' + str(long_es) + ' < ' + str(entr_es + 1))


#Dada una palabra, la codifica según nuestro diccionario.
def codifica(palabra, dic):
    codif = ''
    for letra in palabra:
        codif += dic[letra]
    return codif

medieval_en = codifica('medieval', cod_en)
print('Codificación en inglés: ' + str(medieval_en) + ' de longitud ' + str(len(medieval_en)))
len_bin = len('medieval')*math.ceil(math.log(len(cod_en),2))
print('La longitud en binario usual sería: ' + str(len_bin))
print("La mejora en longitud es de: " + str(len_bin/len(medieval_en)))
medieval_es = codifica('medieval', cod_es)
print('Codificación en español: ' + str(medieval_es) + ' de longitud ' + str(len(medieval_es)))
len_bin = len('medieval')*math.ceil(math.log(len(cod_es),2))
print('La longitud en binario usual sería: ' + str(len_bin))
print("La mejora en longitud es de: " + str(len_bin/len(medieval_es)))


#Dado un codigo binario, decodifica la palabra según nuestro diccionario
def decodifica(palabra, dic):
    #Invertimos el diccionario
    dic_inv = dict(zip(list(dic.values()),list(dic.keys())))
    decodif = ''
    subcadena = ''
    for letra in palabra:
        subcadena += letra
        if subcadena in dic_inv:
            decodif += dic_inv[subcadena]
            subcadena = ''
    
    return decodif

print("10111101101110110111011111 decodificado en inglés es: " + decodifica('10111101101110110111011111', cod_en))
print("10111101101110110111011111 decodificado en español es: " + decodifica('10111101101110110111011111', cod_es))

