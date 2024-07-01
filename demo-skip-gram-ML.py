import io
import re
import string
import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

#%load_ext tensorboard

SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

sentence = "The wide road shimmered in the hot sun"
tokens = list(sentence.lower().split())
print("Fatiando a sentença")
print("Quantas sílabas: ", len(tokens))
print()

vocab, index = {}, 1 # iniciando a indexação com 1
vocab['<pad>'] = 0 # Adicionando o elemento inicial no indice 0

for token in tokens:
    # Verifico se a palavra já está no dicionario
    if token not in vocab:
        vocab[token] = index
        index = index + 1
    
vocab_size = len(vocab)
print(vocab)
print()

# Gerando o vocabulário inverso
inverse_vocab = {index: token for token, index in vocab.items()}
print("Dicionando inverso: ")
print(inverse_vocab)
print()

# Gerando um exemplo de sequencia
example_sequence = [vocab[word] for word in tokens]
print("Exemplo de sequencia")
print(example_sequence)
print()


print(" FIM DA SEPARAÇÃO DAS PALAVRAS, INICIANDO O SKIP-GRAM\n")

print(" Iniciando o skip-gram\n")
window_size = 2
positive_skip_grams,_= tf.keras.preprocessing.sequence.skipgrams(example_sequence, vocabulary_size=vocab_size, window_size=window_size, negative_samples=0)
print(len(positive_skip_grams))

for target, context in positive_skip_grams[:5]:
    print(f"({target}, {context}): ({inverse_vocab[target]}, {inverse_vocab[context]})")

print("Construindo um exemplo de treinamento")

# Obtendo palavras-alvo e de contexto para um skip-gram positivo.
target_word, context_word = positive_skip_grams[0]

# Definindo o número de amostras negativas por contexto positivo.
num_ns = 4

context_class = tf.reshape(tf.constant(context_word, dtype="int64"), (1, 1))
negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
    true_classes=context_class,  #classe que deve ser amostrada como 'positiva'
    num_true=1,  # cada skip-gram positivo tem 1 classe de contexto positiva
    num_sampled=num_ns,  # número de palavras de contexto negativo para amostra
    unique=True,  # todas as amostras negativas devem ser únicas
    range_max=vocab_size,  # escolha o índice das amostras de [0, vocab_size]
    seed=SEED,  # semente para reprodutibilidade
    name="negative_sampling"  # nome desta operação
)
print(negative_sampling_candidates)
print([inverse_vocab[index.numpy()] for index in negative_sampling_candidates])

# Reduzindo uma dimensão para poder usar concatenação (na próxima etapa).
squeezed_context_class = tf.squeeze(context_class, 1)

# Concatenando uma palavra de contexto positiva com palavras de amostra negativas.
context = tf.concat([squeezed_context_class, negative_sampling_candidates], 0)

# Rotulando a primeira palavra do contexto como `1` (positivo) seguida por `num_ns` `0`s (negativo).
label = tf.constant([1] + [0]*num_ns, dtype="int64")
target = target_word

print(f"target_index    : {target}")
print(f"target_word     : {inverse_vocab[target_word]}")
print(f"context_indices : {context}")
print(f"context_words   : {[inverse_vocab[c.numpy()] for c in context]}")
print(f"label           : {label}")

print("target  :", target)
print("context :", context)
print("label   :", label)