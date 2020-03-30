# Forward and LRP pass for the Transformer-CNN solubility model.
# Usage: python3 lrp.py SMILES
# Authors: Dr. Pavel Karpov, Dr. Igor V. Tetko, BIGCHEM GmbH, 2020.
# email: carpovpv@gmail.com

import math
import os
import pickle
import sys
import re

import cairosvg
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from rdkit.Chem import Draw, Descriptors, MolToSmiles, MolFromSmiles, CanonSmiles

# the parameters are the same as for Transformer-CNN model.
N_HIDDEN = 512
N_HIDDEN_CNN = 512
EMBEDDING_SIZE = 64
KEY_SIZE = EMBEDDING_SIZE
CONV_OFFSET = 20

# vocabulary
chars = " ^#%()+-./0123456789=@ABCDEFGHIKLMNOPRSTVXYZ[\\]abcdefgilmnoprstuy$"
vocab_size = len(chars)
char_to_ix = {ch: i for i, ch in enumerate(chars)}

# input
fname_mod = sys.argv[1]
smiles = sys.argv[2]


def tokenize_smiles(smiles):
    pattern = r'#|=|-[0-9]*|\+[0-9]*|[0-9]|\[.{2,5}\]|%[0-9]{2}|\(|\)|\.|/|\\|:|@+|\{|\}|Cl|Ca|Cu|Br|Be|Ba|Bi|' \
              'Si|Se|Sr|Na|Ni|Rb|Ra|Xe|Li|Al|As|Ag|Au|Mg|Mn|Te|Zn|He|Kr|Fe|[BCFHIKNOPScnos]'
    return re.findall(pattern, smiles)


def LRPCheck(label, x, val, verbose=True):
    s = 0.0
    v = np.sum(val)

    if isinstance(x, list):
        for q in x:
            s = s + np.sum(q)
    else:
        s = np.sum(x)

    s = round(s, 7)
    if np.isnan(s):
        print(label, "NaN")
        sys.exit(0)

    if verbose:
        print("{:25}|{:15.5f}  |{:15.5g}  |{:15.5g}%   | ".format(label, s, v - s, (v - s) / v * 100.))


def calcLRPDenseOut(l_previous, w, l_next):
    zij = np.transpose(w[0]) * np.reshape(l_previous, (1, -1))
    zij = zij / (np.sum(zij, axis=1) + w[1])
    R = np.dot(l_next, zij)
    return R


def calcLRPDenseInner(l_previous, w, l_next):
    x = np.reshape(l_previous, (-1, 1))
    q = np.hstack([x for i in range(l_next.shape[0])])

    q = q.astype(np.float64)
    w0 = w[0].astype(np.float64)

    zij = q * w[0]
    z = np.sum(zij, axis=0) + w[1] + 1e-32

    zij = zij / z
    return np.dot(l_next, np.transpose(zij))


def calcLRPAddition(l_first, l_second, l_sum, R):
    result = np.copy(l_sum)

    # to avoid division by zero error if both values are zero
    result[result == 0.0] = 1.0e32

    f = l_first / result

    r_first = R * f
    r_second = R - r_first

    return r_first, r_second


def calcLRPPool(l_embed, inds, R):
    demax = np.zeros((l_embed.shape[0], inds.shape[0]), dtype=np.float32)
    for i in range(inds.shape[0]):
        demax[inds[i], i] = R[i]

    return demax


def calcLRPConv(l_prev, w, l_out):
    y = np.zeros(l_prev.shape, dtype=np.float64)
    for i in range(l_prev.shape[0]):
        x_ = l_prev[i]
        y_ = l_out[i]
        y[i] = calcLRPDenseInner(x_, w, y_)

    return y


def calcLRPConvStride(l_prev, w, l_out, stride):
    y = np.zeros_like(l_prev)
    w_ = np.reshape(w[0], (-1, l_out.shape[1]))
    for i in range(l_prev.shape[0] - stride - 1):
        x_ = l_prev[i:i + stride, :].flatten()
        y_ = l_out[i]

        z = calcLRPDenseInner(x_, [w_, w[1]], y_)
        z = np.reshape(z, (stride, -1))

        y[i:i + stride] = y[i:i + stride] + z

    s = np.sum(y)
    y = y / s * np.sum(l_out)
    return y


# load the model
d = pickle.load(open(fname_mod, "rb"))
info = d[0]
d = d[1]


def calcQSAR(ch, atom, MolWt, doLrp=True, verbose=True):
    mol = MolToSmiles(ch, rootedAtAtom=atom, canonical=False, doRandom=False, isomericSmiles=False)

    N = len(mol)
    NN = N + CONV_OFFSET

    if verbose: print("Analyzing SMILES string: ", mol)

    x = np.zeros(NN, np.int32)
    for i in range(N):
        x[i] = char_to_ix[mol[i]]

    # positional encoding matrix
    pos = np.zeros((NN, EMBEDDING_SIZE), dtype=np.float32)
    for j in range(N):
        for i in range(EMBEDDING_SIZE):
            if i % 2 == 0:
                pos[j, i] = np.sin((j + 1) / np.power(10000.0, i / EMBEDDING_SIZE))
            else:
                pos[j, i] = np.cos((j + 1) / np.power(10000.0, (i - 1) / EMBEDDING_SIZE))

    # mask
    left_mask = np.zeros((NN, NN), dtype=np.float32)
    left_mask[:, :N] = 1

    # the first matrix is our SMILES embeddings
    embed = d[0]
    smiles_embed = np.zeros_like(pos)

    for i in range(NN):
        smiles_embed[i] = embed[x[i]] + pos[i]

    l_embed = smiles_embed

    sa = []
    # next 3*10 matrixes are for SelfAttentions
    for block in range(10):
        K = d[1 + 3 * block]
        V = d[2 + 3 * block]
        Q = d[3 + 3 * block]

        q = np.dot(smiles_embed, Q)
        k = np.dot(smiles_embed, K)
        v = np.dot(smiles_embed, V)

        k = np.transpose(k)
        a = np.dot(q, k) / np.sqrt(EMBEDDING_SIZE)
        a = np.exp(a) * left_mask

        for i in range(a.shape[0]):
            a[i, :] = a[i, :] / np.sum(a[i])

        sa.append(np.dot(a, v))

    # concatenate all self-attention results
    sa = np.concatenate(sa, axis=1)

    # TimeDistributed Dense
    l_dense = np.dot(sa, d[31]) + d[32]

    # residual connection with the input to the block
    l_add = l_dense + l_embed

    # normalization
    gamma = d[33]
    beta = d[34]

    mean = np.mean(l_add, axis=-1, keepdims=True)
    std = np.std(l_add, axis=-1, keepdims=True)
    l_norm = gamma * (l_add - mean) / (std + 1e-6) + beta

    # 1D convolutions
    l_c1 = np.dot(l_norm, d[35][0]) + d[36]
    # relu activation
    l_c1[l_c1 < 0] = 0

    # 2 1D convolution without activation
    l_c2 = np.dot(l_c1, d[37][0]) + d[38]

    # add
    l_ff = l_norm + l_c2

    # normalization
    gamma = d[39]
    beta = d[40]

    mean = np.mean(l_ff, axis=-1, keepdims=True)
    std = np.std(l_ff, axis=-1, keepdims=True)
    l_embed = gamma * (l_ff - mean) / (std + 1e-6) + beta

    #################### 2 block #######################

    sa = []
    # next 3*10 matrixes are for SelfAttentions
    for block in range(10):
        K = d[41 + 3 * block]
        V = d[42 + 3 * block]
        Q = d[43 + 3 * block]

        q = np.dot(l_embed, Q)
        k = np.dot(l_embed, K)
        v = np.dot(l_embed, V)

        k = np.transpose(k)
        a = np.dot(q, k) / np.sqrt(EMBEDDING_SIZE)
        a = np.exp(a) * left_mask

        for i in range(a.shape[0]):
            a[i, :] = a[i, :] / np.sum(a[i])

        sa.append(np.dot(a, v))

    # concatenate all self-attention results
    sa = np.concatenate(sa, axis=1)

    # TimeDistributed Dense
    l_dense = np.dot(sa, d[71]) + d[72]

    # residual connection with the input to the block
    l_add = l_dense + l_embed

    # normalization
    gamma = d[73]
    beta = d[74]

    mean = np.mean(l_add, axis=-1, keepdims=True)
    std = np.std(l_add, axis=-1, keepdims=True)
    l_norm = gamma * (l_add - mean) / (std + 1e-6) + beta

    # 1D convolutions
    l_c1 = np.dot(l_norm, d[75][0]) + d[76]
    # relu activation
    l_c1[l_c1 < 0] = 0

    # 2 1D convolution without activation
    l_c2 = np.dot(l_c1, d[77][0]) + d[78]

    # add
    l_ff = l_norm + l_c2

    # normalization
    gamma = d[79]
    beta = d[80]

    mean = np.mean(l_ff, axis=-1, keepdims=True)
    std = np.std(l_ff, axis=-1, keepdims=True)
    l_embed = gamma * (l_ff - mean) / (std + 1e-6) + beta

    ######################## 3 block ####################

    sa = []
    # next 3*10 matrixes are for SelfAttentions
    for block in range(10):
        K = d[81 + 3 * block]
        V = d[82 + 3 * block]
        Q = d[83 + 3 * block]

        q = np.dot(l_embed, Q)
        k = np.dot(l_embed, K)
        v = np.dot(l_embed, V)

        k = np.transpose(k)
        a = np.dot(q, k) / np.sqrt(EMBEDDING_SIZE)
        a = np.exp(a) * left_mask

        for i in range(a.shape[0]):
            a[i, :] = a[i, :] / np.sum(a[i])

        sa.append(np.dot(a, v))

    # concatenate all self-attention results
    sa = np.concatenate(sa, axis=1)

    # TimeDistributed Dense
    l_dense = np.dot(sa, d[111]) + d[112]

    # residual connection with the input to the block
    l_add = l_dense + l_embed

    # normalization
    gamma = d[113]
    beta = d[114]

    mean = np.mean(l_add, axis=-1, keepdims=True)
    std = np.std(l_add, axis=-1, keepdims=True)
    l_norm = gamma * (l_add - mean) / (std + 1e-6) + beta

    # 1D convolutions
    l_c1 = np.dot(l_norm, d[115][0]) + d[116]

    # relu activation
    l_c1[l_c1 < 0] = 0

    # 2 1D convolution without activation
    l_c2 = np.dot(l_c1, d[117][0]) + d[118]

    # add
    l_ff = l_norm + l_c2

    # normalization
    gamma = d[119]
    beta = d[120]

    mean = np.mean(l_ff, axis=-1, keepdims=True)
    std = np.std(l_ff, axis=-1, keepdims=True)
    l_embed = gamma * (l_ff - mean) / (std + 1e-6) + beta

    l_encoder_out = np.copy(l_embed)
    # end of encoder

    # ============================================

    # 1 convolution
    lc_1 = np.dot(l_embed, d[121]) + d[122]
    lc_1[lc_1 < 0] = 0.0
    max_1 = np.argmax(lc_1, axis=0)
    lc_1 = np.max(lc_1, axis=0)

    # 2/200 (usual valid convolutions)
    w = d[123][:, :].reshape((-1, 200))
    lc_2 = np.zeros((NN - 1, 200), dtype=np.float32)
    for i in range(NN - 1):
        x_ = l_embed[i:i + 2, :].flatten()
        x_ = np.dot(x_, w) + d[124]
        x_[x_ < 0] = 0.0
        lc_2[i] = x_
    max_2 = np.argmax(lc_2, axis=0)
    lc_2 = np.max(lc_2, axis=0)

    # 3/200
    w = d[125][:, :].reshape((-1, 200))
    lc_3 = np.zeros((NN - 2, 200), dtype=np.float32)
    for i in range(NN - 2):
        x_ = l_embed[i:i + 3, :].flatten()
        x_ = np.dot(x_, w) + d[126]
        x_[x_ < 0] = 0.0
        lc_3[i] = x_
    max_3 = np.argmax(lc_3, axis=0)
    lc_3 = np.max(lc_3, axis=0)

    # 4/200
    w = d[127][:, :].reshape((-1, 200))
    lc_4 = np.zeros((NN - 3, 200), dtype=np.float32)
    for i in range(NN - 3):
        x_ = l_embed[i:i + 4, :].flatten()
        x_ = np.dot(x_, w) + d[128]
        x_[x_ < 0] = 0.0
        lc_4[i] = x_
    max_4 = np.argmax(lc_4, axis=0)
    lc_4 = np.max(lc_4, axis=0)

    # 5/200
    w = d[129][:, :].reshape((-1, 200))
    lc_5 = np.zeros((NN - 4, 200), dtype=np.float32)
    for i in range(NN - 4):
        x_ = l_embed[i:i + 5, :].flatten()
        x_ = np.dot(x_, w) + d[130]
        x_[x_ < 0] = 0.0
        lc_5[i] = x_
    max_5 = np.argmax(lc_5, axis=0)
    lc_5 = np.max(lc_5, axis=0)

    # 6/100
    w = d[131][:, :].reshape((-1, 100))
    lc_6 = np.zeros((NN - 5, 100), dtype=np.float32)
    for i in range(NN - 5):
        x_ = l_embed[i:i + 6, :].flatten()
        x_ = np.dot(x_, w) + d[132]
        x_[x_ < 0] = 0.0
        lc_6[i] = x_
    max_6 = np.argmax(lc_6, axis=0)
    lc_6 = np.max(lc_6, axis=0)

    # 7/100
    w = d[133][:, :].reshape((-1, 100))
    lc_7 = np.zeros((NN - 6, 100), dtype=np.float32)
    for i in range(NN - 6):
        x_ = l_embed[i:i + 7, :].flatten()
        x_ = np.dot(x_, w) + d[134]
        x_[x_ < 0] = 0.0
        lc_7[i] = x_
    max_7 = np.argmax(lc_7, axis=0)
    lc_7 = np.max(lc_7, axis=0)

    # 8/100
    w = d[135][:, :].reshape((-1, 100))
    lc_8 = np.zeros((NN - 7, 100), dtype=np.float32)
    for i in range(NN - 7):
        x_ = l_embed[i:i + 8, :].flatten()
        x_ = np.dot(x_, w) + d[136]
        x_[x_ < 0] = 0.0
        lc_8[i] = x_
    max_8 = np.argmax(lc_8, axis=0)
    lc_8 = np.max(lc_8, axis=0)

    # 9/100
    w = d[137][:, :].reshape((-1, 100))
    lc_9 = np.zeros((NN - 8, 100), dtype=np.float32)
    for i in range(NN - 8):
        x_ = l_embed[i:i + 9, :].flatten()
        x_ = np.dot(x_, w) + d[138]
        x_[x_ < 0] = 0.0
        lc_9[i] = x_
    max_9 = np.argmax(lc_9, axis=0)
    lc_9 = np.max(lc_9, axis=0)

    # 10/100
    w = d[139][:, :].reshape((-1, 100))
    lc_10 = np.zeros((NN - 9, 100), dtype=np.float32)
    for i in range(NN - 9):
        x_ = l_embed[i:i + 10, :].flatten()
        x_ = np.dot(x_, w) + d[140]
        x_[x_ < 0] = 0.0
        lc_10[i] = x_
    max_10 = np.argmax(lc_10, axis=0)
    lc_10 = np.max(lc_10, axis=0)

    # 15/160
    w = d[141][:, :].reshape((-1, 160))
    lc_15 = np.zeros((NN - 14, 160), dtype=np.float32)
    for i in range(NN - 14):
        x_ = l_embed[i:i + 15, :].flatten()
        x_ = np.dot(x_, w) + d[142]
        x_[x_ < 0] = 0.0
        lc_15[i] = x_
    max_15 = np.argmax(lc_15, axis=0)
    lc_15 = np.max(lc_15, axis=0)

    # 20/160
    w = d[143][:, :].reshape((-1, 160))
    lc_20 = np.zeros((NN - 19, 160), dtype=np.float32)
    for i in range(NN - 19):
        x_ = l_embed[i:i + 20, :].flatten()
        x_ = np.dot(x_, w) + d[144]
        x_[x_ < 0] = 0.0
        lc_20[i] = x_
    max_20 = np.argmax(lc_20, axis=0)
    lc_20 = np.max(lc_20, axis=0)

    # ============================================
    # concatenate
    l_cnn = np.concatenate([lc_1, lc_2, lc_3, lc_4, lc_5,
                            lc_6, lc_7, lc_8, lc_9, lc_10,
                            lc_15, lc_20])

    l_dense = np.dot(l_cnn, d[145]) + d[146]
    l_dense[l_dense < 0] = 0.0

    # highway
    transform_gate = 1.0 / (1.0 + np.exp(-np.dot(l_dense, d[147]) - d[148]))
    carry_gate = 1.0 - transform_gate

    transformed_data = np.dot(l_dense, d[149]) + d[150]
    transformed_data[transformed_data < 0] = 0.0

    # multiply
    transformed_gated = transform_gate * transformed_data
    identity_gated = carry_gate * l_dense

    # final highway output
    l_highway = transformed_gated + identity_gated

    # the last layer
    l_out = np.dot(l_highway, d[151]) + d[152]

    # for regression linear kernel
    # for classification sigmoid
    if info[1] == "classification":
        l_out = 1.0 / (1.0 + np.exp(-l_out))

    result = l_out[0]
    l_out[0] = eval(info[2])

    y_real = l_out
    if verbose: print("Prognosis:\t", str(l_out[0]) + ", " + info[3], sep="")

    if doLrp == False:
        return l_out[0]

    if verbose: print("\nExplaining the result with LRP technique.\n")
    if verbose: print("   Layer                     Relevance(l)          Delta            Bias(%)\n")

    R_highway = calcLRPDenseOut(l_highway, [d[151], d[152]], l_out)
    LRPCheck("HighWay Output:", R_highway, l_out, verbose)

    R_identity, R_transformed_gated = calcLRPAddition(identity_gated, transformed_gated, l_highway, R_highway)
    # LRPCheck("Identity Gated:", [R_identity, R_transformed_gated], R_highway)

    R_dense_high3 = calcLRPDenseInner(l_dense, [d[149], d[150]], R_transformed_gated)
    R_input_highway = R_identity + R_dense_high3  # + R_dense_high21 + R_dense_high22 + R_dense_high3

    LRPCheck("Input HighWay:", R_input_highway, R_highway, verbose)

    R_cnn = calcLRPDenseInner(l_cnn, [d[145], d[146]], R_input_highway)
    # LRPCheck("CNN concat:", R_cnn, l_out)

    # concatenation
    R_cnn_1 = R_cnn[:100]
    R_cnn_2 = R_cnn[100:300]
    R_cnn_3 = R_cnn[300:500]
    R_cnn_4 = R_cnn[500:700]
    R_cnn_5 = R_cnn[700:900]
    R_cnn_6 = R_cnn[900:1000]
    R_cnn_7 = R_cnn[1000:1100]
    R_cnn_8 = R_cnn[1100:1200]
    R_cnn_9 = R_cnn[1200:1300]
    R_cnn_10 = R_cnn[1300:1400]
    R_cnn_15 = R_cnn[1400:1560]
    R_cnn_20 = R_cnn[1560:]

    # Increase the dimension pulling the relevance to a maximum descriptor.
    d_1 = calcLRPPool(l_embed, max_1, R_cnn_1)
    d_2 = calcLRPPool(l_embed, max_2, R_cnn_2)
    d_3 = calcLRPPool(l_embed, max_3, R_cnn_3)
    d_4 = calcLRPPool(l_embed, max_4, R_cnn_4)
    d_5 = calcLRPPool(l_embed, max_5, R_cnn_5)
    d_6 = calcLRPPool(l_embed, max_6, R_cnn_6)
    d_7 = calcLRPPool(l_embed, max_7, R_cnn_7)
    d_8 = calcLRPPool(l_embed, max_8, R_cnn_8)
    d_9 = calcLRPPool(l_embed, max_9, R_cnn_9)
    d_10 = calcLRPPool(l_embed, max_10, R_cnn_10)
    d_15 = calcLRPPool(l_embed, max_15, R_cnn_15)
    d_20 = calcLRPPool(l_embed, max_20, R_cnn_20)

    LRPCheck("DeMaxPool:", [d_1, d_2, d_3, d_4, d_5, d_6, d_7, d_8, d_9, d_10, d_15, d_20], R_input_highway, verbose)

    if verbose: print("Char-CNN block:")

    R_cnn_1 = calcLRPConv(l_embed, [d[121], d[122]], d_1)
    LRPCheck("  Conv1:", R_cnn_1, np.sum(d_1), verbose)

    R_cnn_2 = calcLRPConvStride(l_embed, [d[123], d[124]], d_2, 2)
    LRPCheck("  Conv2:", R_cnn_2, np.sum(d_2), verbose)

    R_cnn_3 = calcLRPConvStride(l_embed, [d[125], d[126]], d_3, 3)
    LRPCheck("  Conv3:", R_cnn_3, np.sum(d_3), verbose)

    R_cnn_4 = calcLRPConvStride(l_embed, [d[127], d[128]], d_4, 4)
    LRPCheck("  Conv4:", R_cnn_4, np.sum(d_4), verbose)

    R_cnn_5 = calcLRPConvStride(l_embed, [d[129], d[130]], d_5, 5)
    LRPCheck("  Conv5:", R_cnn_5, np.sum(d_5), verbose)

    R_cnn_6 = calcLRPConvStride(l_embed, [d[131], d[132]], d_6, 6)
    LRPCheck("  Conv6:", R_cnn_6, np.sum(d_6), verbose)

    R_cnn_7 = calcLRPConvStride(l_embed, [d[133], d[134]], d_7, 7)
    LRPCheck("  Conv7:", R_cnn_7, np.sum(d_7), verbose)

    R_cnn_8 = calcLRPConvStride(l_embed, [d[135], d[136]], d_8, 8)
    LRPCheck("  Conv8:", R_cnn_8, np.sum(d_8), verbose)

    R_cnn_9 = calcLRPConvStride(l_embed, [d[137], d[138]], d_9, 9)
    LRPCheck("  Conv9:", R_cnn_9, np.sum(d_9), verbose)

    R_cnn_10 = calcLRPConvStride(l_embed, [d[139], d[140]], d_10, 10)
    LRPCheck("  Conv10:", R_cnn_10, np.sum(d_10), verbose)

    R_cnn_15 = calcLRPConvStride(l_embed, [d[141], d[142]], d_15, 15)
    LRPCheck("  Conv15:", R_cnn_15, np.sum(d_15), verbose)

    R_cnn_20 = calcLRPConvStride(l_embed, [d[143], d[144]], d_20, 20)
    LRPCheck("  Conv20:", R_cnn_20, np.sum(d_20), verbose)

    R_cnn = R_cnn_1 + R_cnn_2 + R_cnn_3 + R_cnn_4 + R_cnn_5 + R_cnn_6 + \
            R_cnn_7 + R_cnn_8 + R_cnn_9 + R_cnn_10 + R_cnn_15 + R_cnn_20

    LRPCheck("Deconvolution:", R_cnn, l_out, verbose)

    scores = np.sum(R_cnn, axis=1)

    return y_real[0], scores, np.sum(l_out) - np.sum(R_cnn)


# Main Code
smiles = CanonSmiles(smiles, useChiral=0)
mol = MolFromSmiles(smiles)
mw = Descriptors.ExactMolWt(mol)
atoms = {a.GetIdx(): a.GetSmarts() for a in mol.GetAtoms()}
impacts = np.zeros(len(atoms), dtype='float')

print("Predicting %i atoms..." % (len(atoms)))
vals = []
for idx, a in tqdm(atoms.items()):
    val, scores, _ = calcQSAR(mol, idx, mw, verbose=False)
    vals.append(val)
    impacts[idx] = scores[0]

res = np.mean(vals)
std = np.std(vals)

print("\n{} Prediction = {:.7f} +/- {:7f} {}".format(info[0], res, 1.96 * std / math.sqrt(len(vals)), info[3]))

# plot the results
y_min = np.min(impacts)
y_max = np.max(impacts)
diff = y_max - y_min

x_vals = tokenize_smiles(smiles)
y_vals = list()
char_colors = list()
mol_cols = dict()

k = 0
p = 0
if info[0] == 'AMES':
    p = 1

for i, s in enumerate(tokenize_smiles(smiles)):
    triple = [0, 0, 0]
    n = ""
    if s == atoms[k]:
        y_vals.append(impacts[k])
        if impacts[k] > y_max / 10:
            triple[1-p] = 1 - 0.66 * (impacts[k] / y_max)
        elif impacts[k] < y_min / 10:
            triple[abs(0-p)] = 1 - 0.66 * (impacts[k] / y_min)
        else:
            triple = [1, 1, 1]
        char_colors.append(tuple(triple))
        mol_cols[k] = tuple(triple)
        if k < len(atoms) - 1:  # if special character at last place in SMILES
            k += 1
    else:
        y_vals.append(0.)
        char_colors.append((0., 0., 0.))


# draw highlighted structure
Draw.rdMolDraw2D.PrepareMolForDrawing(mol, addChiralHs=False)
drawer = Draw.rdMolDraw2D.MolDraw2DSVG(500, 500)
drawer.DrawMolecule(mol, highlightAtoms=atoms.keys(), highlightBonds=[], highlightAtomColors=mol_cols)
drawer.FinishDrawing()
svg = drawer.GetDrawingText().replace('svg:', '')
with open("mol.svg", "w") as f:
    f.write(svg)
cairosvg.svg2png(url='mol.svg', write_to="mol.png")  # convert to PNG for mpl

# final plot
img = plt.imread('mol.png')
os.remove('mol.png')
os.remove('mol.svg')
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
for i, y in enumerate(y_vals):
    axs[0].bar(i, y, color=char_colors[i])
axs[0].grid(True)
axs[0].set_xticks(range(len(x_vals)))
axs[0].set_xticklabels(x_vals)
axs[0].set_xlim([-1, len(x_vals)])
axs[0].set_ylabel('Prediction Score')
axs[1].imshow(img)
axs[1].axis('off')
text = "{} Prediction = {:.7f} +/- {:7f} {}".format(info[0], res, 1.96 * std / math.sqrt(len(vals)), info[3])
axs[1].text(0.5, 0., text, transform=axs[1].transAxes, horizontalalignment='center',
            bbox={'facecolor': 'gray', 'alpha': 0.25, 'pad': 10})
fig.suptitle(fname_mod.split('/')[-1].split('.')[0].upper())
plt.savefig('output.png')

print("\nAll done! --> Check 'output.png'")
