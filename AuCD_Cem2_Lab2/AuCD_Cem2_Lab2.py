import numpy as np
from PIL import Image
from math import cos, pi, sqrt
import matplotlib.pyplot as plt

def png_to_raw(input_path, output_path):

    img = Image.open(input_path)
   
    img_array = np.array(img)
    with open(output_path, 'wb') as f:
        f.write(img_array.tobytes())
    
    print(f"Сохранено RAW изображение: {output_path}")
    print(f"Размеры: {img_array.shape}, Тип данных: {img_array.dtype}")

def raw_to_png(input_path, output_path, width, height, channels=3, dtype=np.uint8):
    """
    :param width: Ширина изображения
    :param height: Высота изображения
    :param channels: Количество каналов (3 для RGB, 1 для grayscale)
    :param dtype: Тип данных (np.uint8 для 8-битных изображений)
    """

    with open(input_path, 'rb') as f:
        raw_data = f.read()
    
    img_array = np.frombuffer(raw_data, dtype=dtype)
    
    if channels == 1:
        img_array = img_array.reshape((height, width))
    else:
        img_array = img_array.reshape((height, width, channels))
    
    img = Image.fromarray(img_array)
    img.save(output_path)
    print(f"Сохранено PNG изображение: {output_path}")

def rgb_to_ycbcr(r, g, b):
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128
    return np.clip(y, 0, 255), np.clip(cb, 0, 255), np.clip(cr, 0, 255)

def ycbcr_to_rgb(y, cb, cr):
    cb_shifted = cb - 128
    cr_shifted = cr - 128

    r = y + 1.402 * cr_shifted
    g = y - 0.344136 * cb_shifted - 0.714136 * cr_shifted
    b = y + 1.772 * cb_shifted

    return np.clip(r, 0, 255), np.clip(g, 0, 255), np.clip(b, 0, 255)

def raw_to_ycbcr(raw_file, width, height):
    with open(raw_file, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    
    expected_size = width * height * 3
    actual_size = data.size
    
    if actual_size != expected_size:
        if actual_size > expected_size:
            data = data[:expected_size]
        else:
            raise ValueError(f"Недостаточно данных. Ожидалось {expected_size} байт, получено {actual_size} байт")
    
    rgb = data.reshape((height, width, 3))
    
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    
    y, cb, cr = rgb_to_ycbcr(r, g, b)
    return y, cb, cr

def ycbcr_to_raw(y, cb, cr, output_file):
    r, g, b = ycbcr_to_rgb(y, cb, cr)
    
    r = np.clip(r, 0, 255).astype(np.uint8)
    g = np.clip(g, 0, 255).astype(np.uint8)
    b = np.clip(b, 0, 255).astype(np.uint8)
    
    rgb = np.stack((r, g, b), axis=2)
    
    with open(output_file, 'wb') as f:
        f.write(rgb.tobytes())

def downsample_image(channel, scale_factor=2):
    height, width = channel.shape
    new_width = width // scale_factor
    new_height = height // scale_factor
    downsampled_channel = np.zeros((new_height, new_width), dtype=channel.dtype)

    for y in range(new_height):
        for x in range(new_width):
            original_x = x * scale_factor
            original_y = y * scale_factor
            #downsampled_channel[y, x] = channel[original_y, original_x]
            # Используем усреднение для downsampling
            downsampled_channel[y, x] = np.mean(
                channel[original_y:original_y + scale_factor, original_x:original_x + scale_factor])

    return downsampled_channel

def upsample_image(channel, scale_factor=2):
    new_height, new_width = channel.shape
    height = new_height * scale_factor
    width = new_width * scale_factor
    upsampled_channel = np.zeros((height, width), dtype=channel.dtype)

    for y in range(new_height):
        for x in range(new_width):
            upsampled_channel[y * scale_factor:(y + 1) * scale_factor,
                              x * scale_factor:(x + 1) * scale_factor] = channel[y, x]

    return upsampled_channel

def splitting_img(N=8, img=(0,0)):
    h, w = img.shape
    h_blocks = (h + N - 1) // N
    w_blocks = (w + N - 1) // N
    
    blocks = np.zeros((h_blocks * w_blocks, N, N), dtype=np.uint8)
    
    for block_idx in range(h_blocks * w_blocks):
        y = (block_idx // w_blocks) * N
        x = (block_idx % w_blocks) * N
        
        y_end = min(y + N, h)
        x_end = min(x + N, w)
        
        blocks[block_idx, :y_end-y, :x_end-x] = img[y:y_end, x:x_end]
        
    return blocks

def zigzag(matrix):
    n = len(matrix)
    result = []
    
    for s in range(0, 2*n-1):
        if s < n:
            if s % 2 == 0:
                i, j = s, 0
                while i >= 0 and j < n:
                    result.append(matrix[i][j])
                    i -= 1
                    j += 1
            else:
                i, j = 0, s
                while j >= 0 and i < n:
                    result.append(matrix[i][j])
                    i += 1
                    j -= 1
        else:
            if s % 2 == 0:
                i, j = n-1, s-n+1
                while i >= 0 and j < n:
                    result.append(matrix[i][j])
                    i -= 1
                    j += 1
            else:
                i, j = s-n+1, n-1
                while j >= 0 and i < n:
                    result.append(matrix[i][j])
                    i += 1
                    j -= 1
                    
    return result

def precompute_dct_coeffs(N):
    coeffs = np.zeros((N, N, N, N))
    scale = np.ones((N, N))
    
    sqrt2 = sqrt(2)
    for p in range(N):
        for q in range(N):
            if p == 0:
                scale[p,q] /= sqrt2
            if q == 0:
                scale[p,q] /= sqrt2
            for m in range(N):
                for n in range(N):
                    coeffs[p,q,m,n] = cos(pi*p*(2*m+1)/(2*N)) * cos(pi*q*(2*n+1)/(2*N))
    
    return coeffs, scale

def DCT(A):
    N = A.shape[0]
    coeffs, scale = precompute_dct_coeffs(N)
    B = np.zeros((N, N))
    
    for p in range(N):
        for q in range(N):
            total = 0.0
            for m in range(N):
                for n in range(N):
                    total += A[m,n] * coeffs[p,q,m,n]
            B[p,q] = total * scale[p,q] * (2/N)
    
    return B.round().astype(np.int16)

def iDCT(B):
    N = B.shape[0]
    coeffs, scale = precompute_dct_coeffs(N)
    A = np.zeros((N, N))
    
    for m in range(N):
        for n in range(N):
            total = 0.0
            for p in range(N):
                for q in range(N):
                    total += B[p,q] * coeffs[p,q,m,n] * scale[p,q]
            A[m,n] = total * (2/N)
    
    return A.round().astype(np.int16) 

Q50 = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=np.float32)

QCH = np.array([
[17, 18, 24, 47, 99, 99, 99, 99],
[18, 21, 26, 66, 99, 99, 99, 99],
[24, 26, 56, 99, 99, 99, 99, 99],
[47, 66, 99, 99, 99, 99, 99, 99],
[99, 99, 99, 99, 99, 99, 99, 99],
[99, 99, 99, 99, 99, 99, 99, 99],
[99, 99, 99, 99, 99, 99, 99, 99],
[99, 99, 99, 99, 99, 99, 99, 99]    
], dtype=np.float32)

def get_quantization_matrix(qf, is_luma=True):
    if qf <= 0: qf = 1
    if qf > 100: qf = 100
    
    base_matrix = Q50 if is_luma else QCH
    
    if qf < 50:
        scale = 5000 / qf  
    else:
        scale = 200 - 2*qf
    
    Q = np.floor((base_matrix * scale + 50) / 100)
    Q = np.clip(Q, 1, 255)
    return Q.astype(np.uint16)

def quantize(block, Q):
    Q = np.maximum(Q, 1)
    return np.round(block / Q).astype(np.int16)

def dequantize(block, Q):
    result = block * Q
    return np.clip(result, -32768, 32767).astype(np.int16)

def block_quant(block, qf):
    Q = get_quantization_matrix(qf)
    quant_block = quantize(block, Q)
    return quant_block.astype(np.int16)

def encode_dc(dc_coeffs):
    deltas = []
    prev_dc = 0
    for dc in dc_coeffs:
        delta = dc - prev_dc
        deltas.append(delta)
        prev_dc = dc
    return deltas

def decode_dc(delta_coeffs):
    dc_coeffs = []
    prev_dc = 0
    for delta in delta_coeffs:
        dc = prev_dc + delta
        dc_coeffs.append(dc)
        prev_dc = dc
    return dc_coeffs

def rle_encode(block_1d):
    encoded = []
    zero_count = 0
    
    for coeff in block_1d[1:]:
        if coeff == 0:
            zero_count += 1
            if zero_count == 16:
                encoded.append((15, 0))
                zero_count = 0
        else:
            while zero_count >= 16:
                encoded.append((15, 0))
                zero_count -= 16
            encoded.append((zero_count, coeff))
            zero_count = 0
    
    if encoded and encoded[-1] != (0, 0):
        encoded.append((0, 0))
    
    return encoded

def apply_rle_to_blocks(all_blocks_1d, block_size=64):
    rle_encoded = []
    for i in range(0, len(all_blocks_1d), block_size):
        block = all_blocks_1d[i:i+block_size]
        rle_encoded.extend(rle_encode(block))
    return rle_encoded

DC_LUMA_HUFFMAN = {
    0: '00',
    1: '010',
    2: '011',
    3: '100',
    4: '101',
    5: '110',
    6: '1110',
    7: '11110',
    8: '111110',
    9: '1111110',
    10: '11111110',
    11: '111111110'
}
DC_CHROM_HUFFMAN = {
    0: '00',
    1: '01',
    2: '10',
    3: '110',
    4: '1110',
    5: '11110',
    6: '111110',
    7: '1111110',
    8: '11111110',
    9: '111111110',
    10: '1111111110',
    11: '11111111110'
}
AC_LUMA_HUFFMAN = {
    (0,0) : '1010',
    (0,1) : '00',
    (0,2) : '01',
    (0,3) : '100',
    (0,4) : '1011',
    (0,5) : '11010',
    (0,6) : '1111000',
    (0,7) : '11111000',
    (0,8) : '1111110110',
    (0,9) : '1111111110000010',
    (0,10) : '1111111110000011',
    (1,1) : '1100',
    (1,2) : '11011',
    (1,3) : '1111001',
    (1,4) : '111110110',
    (1,5) : '11111110110',
    (1,6) : '1111111110000100',
    (1,7) : '1111111110000101',
    (1,8) : '1111111110000110',
    (1,9) : '1111111110000111',
    (1,10) : '1111111110001000',
    (2,1) : '11100',
    (2,2) : '11111001',
    (2,3) : '1111110111',
    (2,4) : '111111110100',
    (2,5) : '1111111110001001',
    (2,6) : '1111111110001010',
    (2,7) : '1111111110001011',
    (2,8) : '1111111110001100',
    (2,9) : '1111111110001101',
    (2,10) : '1111111110001110',
    (3,1) : '111010',
    (3,2) : '111110111',
    (3,3) : '111111110101',
    (3,4) : '1111111110001111',
    (3,5) : '1111111110010000',
    (3,6) : '1111111110010001',
    (3,7) : '1111111110010010',
    (3,8) : '1111111110010011',
    (3,9) : '1111111110010100',
    (3,10) : '1111111110010101',
    (4,1) : '111011',
    (4,2) : '1111111000',
    (4,3) : '1111111110010110',
    (4,4) : '1111111110010111',
    (4,5) : '1111111110011000',
    (4,6) : '1111111110011001',
    (4,7) : '1111111110011010',
    (4,8) : '1111111110011011',
    (4,9) : '1111111110011100',
    (4,10) : '1111111110011101',
    (5,1) : '1111010',
    (5,2) : '11111110111',
    (5,3) : '1111111110011110',
    (5,4) : '1111111110011111',
    (5,5) : '1111111110100000',
    (5,6) : '1111111110100001',
    (5,7) : '1111111110100010',
    (5,8) : '1111111110100011',
    (5,9) : '1111111110100100',
    (5,10) : '1111111110100101',
    (6,1) : '1111011',
    (6,2) : '111111110110',
    (6,3) : '1111111110100110',
    (6,4) : '1111111110100111',
    (6,5) : '1111111110101000',
    (6,6) : '1111111110101001',
    (6,7) : '1111111110101010',
    (6,8) : '1111111110101011',
    (6,9) : '1111111110101100',
    (6,10) : '1111111110101101',
    (7,1) : '11111010',
    (7,2) : '111111110111',
    (7,3) : '1111111110101110',
    (7,4) : '1111111110101111',
    (7,5) : '1111111110110000',
    (7,6) : '1111111110110001',
    (7,7) : '1111111110110010',
    (7,8) : '1111111110110011',
    (7,9) : '1111111110110100',
    (7,10) : '1111111110110101',
    (8,1) : '111111000',
    (8,2) : '111111111000000',
    (8,3) : '1111111110110110',
    (8,4) : '1111111110110111',
    (8,5) : '1111111110111000',
    (8,6) : '1111111110111001',
    (8,7) : '1111111110111010',
    (8,8) : '1111111110111011',
    (8,9) : '1111111110111100',
    (8,10) : '1111111110111101',
    (9,1) : '111111001',
    (9,2) : '1111111110111110',
    (9,3) : '1111111110111111',
    (9,4) : '1111111111000000',
    (9,5) : '1111111111000001',
    (9,6) : '1111111111000010',
    (9,7) : '1111111111000011',
    (9,8) : '1111111111000100',
    (9,9) : '1111111111000101',
    (9,10) : '1111111111000110',
    (10,1) : '111111010',
    (10,2) : '1111111111000111',
    (10,3) : '1111111111001000',
    (10,4) : '1111111111001001',
    (10,5) : '1111111111001010',
    (10,6) : '1111111111001011',
    (10,7) : '1111111111001100',
    (10,8) : '1111111111001101',
    (10,9) : '1111111111001110',
    (10,10) : '1111111111001111',
    (11,1) : '1111111001',
    (11,2) : '1111111111010000',
    (11,3) : '1111111111010001',
    (11,4) : '1111111111010010',
    (11,5) : '1111111111010011',
    (11,6) : '1111111111010100',
    (11,7) : '1111111111010101',
    (11,8) : '1111111111010110',
    (11,9) : '1111111111010111',
    (11,10) : '1111111111011000',
    (12,1) : '1111111010',
    (12,2) : '1111111111011001',
    (12,3) : '1111111111011010',
    (12,4) : '1111111111011011',
    (12,5) : '1111111111011100',
    (12,6) : '1111111111011101',
    (12,7) : '1111111111011110',
    (12,8) : '1111111111011111',
    (12,9) : '1111111111100000',
    (12,10) : '1111111111100001',
    (13,1) : '11111111000',
    (13,2) : '1111111111100010',
    (13,3) : '1111111111100011',
    (13,4) : '1111111111100100',
    (13,5) : '1111111111100101',
    (13,6) : '1111111111100110',
    (13,7) : '1111111111100111',
    (13,8) : '1111111111101000',
    (13,9) : '1111111111101001',
    (13,10) : '1111111111101010',
    (14,1) : '1111111111101011',
    (14,2) : '1111111111101100',
    (14,3) : '1111111111101101',
    (14,4) : '1111111111101110',
    (14,5) : '1111111111101111',
    (14,6) : '1111111111110000',
    (14,7) : '1111111111110001',
    (14,8) : '1111111111110010',
    (14,9) : '1111111111110011',
    (14,10) : '1111111111110100',
    (15,0) : '11111111001',
    (15,1) : '1111111111110101',
    (15,2) : '1111111111110110',
    (15,3) : '1111111111110111',
    (15,4) : '1111111111111000',
    (15,5) : '1111111111111001',
    (15,6) : '1111111111111010',
    (15,7) : '1111111111111011',
    (15,8) : '1111111111111100',
    (15,9) : '1111111111111101',
    (15,10) : '1111111111111110'
}
AC_CHROM_HUFFMAN = {
    (0,0) : '00',
    (0,1) : '01',
    (0,2) : '100',
    (0,3) : '1010',
    (0,4) : '11000',
    (0,5) : '11001',
    (0,6) : '111000',
    (0,7) : '1111000',
    (0,8) : '111110100',
    (0,9) : '1111110110',
    (0,10) : '111111110100',
    (1,1) : '1011',
    (1,2) : '111001',
    (1,3) : '11110110',
    (1,4) : '111110101',
    (1,5) : '11111110110',
    (1,6) : '111111110101',
    (1,7) : '1111111110001000',
    (1,8) : '1111111110001001',
    (1,9) : '1111111110001010',
    (1,10) : '1111111110001011',
    (2,1) : '11010',
    (2,2) : '11110111',
    (2,3) : '1111110111',
    (2,4) : '111111110110',
    (2,5) : '111111111000010',
    (2,6) : '1111111110001100',
    (2,7) : '1111111110001101',
    (2,8) : '1111111110001110',
    (2,9) : '1111111110001111',
    (2,10) : '1111111110010000',
    (3,1) : '11011',
    (3,2) : '11111000',
    (3,3) : '1111111000',
    (3,4) : '111111110111',
    (3,5) : '1111111110010001',
    (3,6) : '1111111110010010',
    (3,7) : '1111111110010011',
    (3,8) : '1111111110010100',
    (3,9) : '1111111110010101',
    (3,10) : '1111111110010110',
    (4,1) : '111010',
    (4,2) : '111110110',
    (4,3) : '1111111110010111',
    (4,4) : '1111111110011000',
    (4,5) : '1111111110011001',
    (4,6) : '1111111110011010',
    (4,7) : '1111111110011011',
    (4,8) : '1111111110011100',
    (4,9) : '1111111110011101',
    (4,10) : '1111111110011110',
    (5,1) : '111011',
    (5,2) : '1111111001',
    (5,3) : '1111111110011111',
    (5,4) : '1111111110100000',
    (5,5) : '1111111110100001',
    (5,6) : '1111111110100010',
    (5,7) : '1111111110100011',
    (5,8) : '1111111110100100',
    (5,9) : '1111111110100101',
    (5,10) : '1111111110100110',
    (6,1) : '1111001',
    (6,2) : '11111110111',
    (6,3) : '1111111110100111',
    (6,4) : '1111111110101000',
    (6,5) : '1111111110101001',
    (6,6) : '1111111110101010',
    (6,7) : '1111111110101011',
    (6,8) : '1111111110101100',
    (6,9) : '1111111110101101',
    (6,10) : '1111111110101110',
    (7,1) : '1111010',
    (7,2) : '11111111000',
    (7,3) : '1111111110101111',
    (7,4) : '1111111110110000',
    (7,5) : '1111111110110001',
    (7,6) : '1111111110110010',
    (7,7) : '1111111110110011',
    (7,8) : '1111111110110100',
    (7,9) : '1111111110110101',
    (7,10) : '1111111110110110',
    (8,1) : '11111001',
    (8,2) : '1111111110110111',
    (8,3) : '1111111110111000',
    (8,4) : '1111111110111001',
    (8,5) : '1111111110111010',
    (8,6) : '1111111110111011',
    (8,7) : '1111111110111100',
    (8,8) : '1111111110111101',
    (8,9) : '1111111110111110',
    (8,10) : '1111111110111111',
    (9,1) : '111110111',
    (9,2) : '1111111111000000',
    (9,3) : '1111111111000001',
    (9,4) : '1111111111000010',
    (9,5) : '1111111111000011',
    (9,6) : '1111111111000100',
    (9,7) : '1111111111000101',
    (9,8) : '1111111111000110',
    (9,9) : '1111111111000111',
    (9,10) : '1111111111001000',
    (10,1) : '111111000',
    (10,2) : '1111111111001001',
    (10,3) : '1111111111001010',
    (10,4) : '1111111111001011',
    (10,5) : '1111111111001100',
    (10,6) : '1111111111001101',
    (10,7) : '1111111111001110',
    (10,8) : '1111111111001111',
    (10,9) : '1111111111010000',
    (10,10) : '1111111111010001',
    (11,1) : '111111001',
    (11,2) : '1111111111010010',
    (11,3) : '1111111111010011',
    (11,4) : '1111111111010100',
    (11,5) : '1111111111010101',
    (11,6) : '1111111111010110',
    (11,7) : '1111111111010111',
    (11,8) : '1111111111011000',
    (11,9) : '1111111111011001',
    (11,10) : '1111111111011010',
    (12,1) : '111111010',
    (12,2) : '1111111111011011',
    (12,3) : '1111111111011100',
    (12,4) : '1111111111011101',
    (12,5) : '1111111111011110',
    (12,6) : '1111111111011111',
    (12,7) : '1111111111100000',
    (12,8) : '1111111111100001',
    (12,9) : '1111111111100010',
    (12,10) : '1111111111100011',
    (13,1) : '11111111001',
    (13,2) : '1111111111100100',
    (13,3) : '1111111111100101',
    (13,4) : '1111111111100110',
    (13,5) : '1111111111100111',
    (13,6) : '1111111111101000',
    (13,7) : '1111111111101001',
    (13,8) : '1111111111101010',
    (13,9) : '1111111111101011',
    (13,10) : '1111111111101100',
    (14,1) : '11111111100000',
    (14,2) : '1111111111101101',
    (14,3) : '1111111111101110',
    (14,4) : '1111111111101111',
    (14,5) : '1111111111110000',
    (14,6) : '1111111111110001',
    (14,7) : '1111111111110010',
    (14,8) : '1111111111110011',
    (14,9) : '1111111111110100',
    (14,10) : '1111111111110101',
    (15,0) : '1111111010',
    (15,1) : '111111111000011',
    (15,2) : '1111111111110110',
    (15,3) : '1111111111110111',
    (15,4) : '1111111111111000',
    (15,5) : '1111111111111001',
    (15,6) : '1111111111111010',
    (15,7) : '1111111111111011',
    (15,8) : '1111111111111100',
    (15,9) : '1111111111111101',
    (15,10) : '1111111111111110'
}

def encode_dc_with_huffman(delta, flag):
    l, bits = variable_encoding(delta)
    if flag==0:
        huffman_code = DC_LUMA_HUFFMAN[l]
    else:
        huffman_code = DC_CHROM_HUFFMAN[l]
    return huffman_code + bits 

def encode_ac_with_huffman(run, size, value, flag):
    if run >= 16:
        zrl_count = run // 16
        remaining_run = run % 16
        huffman_table = AC_LUMA_HUFFMAN if flag == 0 else AC_CHROM_HUFFMAN
        zrl_code = huffman_table[(15,0)] * zrl_count
        main_code = encode_ac_with_huffman(remaining_run, size, value, flag)
        return zrl_code + main_code

    if size == 0:
        if run == 0:
            return AC_LUMA_HUFFMAN[(0,0)] if flag == 0 else AC_CHROM_HUFFMAN[(0,0)]
        elif run == 15:
            return AC_LUMA_HUFFMAN[(15,0)] if flag == 0 else AC_CHROM_HUFFMAN[(15,0)]
        else:
            raise ValueError(f"Invalid (RUNLENGTH, SIZE): ({run}, {size}). SIZE=0 allowed only for (0,0) or (15,0)")

    huffman_table = AC_LUMA_HUFFMAN if flag == 0 else AC_CHROM_HUFFMAN
    if (run, size) not in huffman_table:
        raise ValueError(f"Неизвестная комбинация (RUNLENGTH, SIZE): ({run}, {size}). Проверьте таблицы Хаффмана")

    if value < 0:
        value_int = value if isinstance(value, int) else int(value)
        bits = bin(value_int & ((1 << size) - 1))[2:].zfill(size)
    else:
        value_int = value if isinstance(value, int) else int(value)
        bits = bin(value_int)[2:].zfill(size)

    return huffman_table[(run, size)] + bits

def jpeg_compress_pipeline(input_path, output_path, qf=90, N=512):
    y, cb, cr = raw_to_ycbcr(input_path, N, N)
    
    y_c, cb_c, cr_c = y, downsample_image(cb), downsample_image(cr)
    print(y_c.shape, cb_c.shape, cr_c.shape)

    components_data = {}
    for name, component in zip(['Y', 'Cb', 'Cr'], [y_c, cb_c, cr_c]):
        blocks = splitting_img(8, component)
        
        dct_blocks = np.array([DCT(block) for block in blocks])
        quant_blocks = np.array([quantize(block, get_quantization_matrix(qf)) for block in dct_blocks])
        
        flag = 0 if name=='Y' else 1

        dc_coeffs = [block[0,0] for block in quant_blocks]
        delta_dc = encode_dc(dc_coeffs)
        dc_bits = [encode_dc_with_huffman(delta, flag) for delta in delta_dc]
        dc_bitstream = ''.join(dc_bits)
        
        ac_bits = []
        for block in quant_blocks:
            zz = zigzag(block)
            rle = rle_encode(zz[1:])

            if not (len(rle) > 0 and rle[-1] == (0,0)):
                rle.append((0,0))
            
            for run_length, value in rle:
                size = 0 if value == 0 else int(np.floor(np.log2(abs(value))) + 1)
                try:
                    ac_bits.append(encode_ac_with_huffman(run_length, size, value, flag))
                except ValueError as e:
                    print(f"Ошибка кодирования блока: {e}")
                    continue
        
        ac_bitstream = ''.join(ac_bits)
        
        components_data[name] = {
            'dc': dc_bitstream,
            'ac': ac_bitstream
        }
    metadata = {
        'width': N,
        'height': N,
        'qf': qf,
        'components': {
            'Y': {
                'dc_length': len(components_data['Y']['dc']),
                'ac_length': len(components_data['Y']['ac'])
            },
            'Cb': {
                'dc_length': len(components_data['Cb']['dc']),
                'ac_length': len(components_data['Cb']['ac'])
            },
            'Cr': {
                'dc_length': len(components_data['Cr']['dc']),
                'ac_length': len(components_data['Cr']['ac'])
            }
        }
    }

    with open(output_path, 'wb') as f:
        f.write(metadata['width'].to_bytes(4, byteorder='big'))
        f.write(metadata['height'].to_bytes(4, byteorder='big'))
        f.write(metadata['qf'].to_bytes(1, byteorder='big'))
        
        for name in ['Y', 'Cb', 'Cr']:
            f.write(metadata['components'][name]['dc_length'].to_bytes(4, byteorder='big'))
            f.write(metadata['components'][name]['ac_length'].to_bytes(4, byteorder='big'))
        
        for name in ['Y', 'Cb', 'Cr']:
            dc_bitstream = components_data[name]['dc']
            dc_padded = dc_bitstream + '0' * ((8 - len(dc_bitstream) % 8) % 8)
            dc_bytes = bytes(int(dc_padded[i:i+8], 2) for i in range(0, len(dc_padded), 8))
            f.write(dc_bytes)
            
            ac_bitstream = components_data[name]['ac']
            ac_padded = ac_bitstream + '0' * ((8 - len(ac_bitstream) % 8) % 8)
            ac_bytes = bytes(int(ac_padded[i:i+8], 2) for i in range(0, len(ac_padded), 8))
            f.write(ac_bytes)
    return components_data

def read_compressed_file(input_path):
    with open(input_path, 'rb') as f:
        width = int.from_bytes(f.read(4), byteorder='big')
        height = int.from_bytes(f.read(4), byteorder='big')
        qf = int.from_bytes(f.read(1), byteorder='big')
        
        lengths = {}
        for name in ['Y', 'Cb', 'Cr']:
            lengths[name] = {
                'dc_length': int.from_bytes(f.read(4), byteorder='big'),
                'ac_length': int.from_bytes(f.read(4), byteorder='big')
            }
        
        all_data = f.read()
    
    full_bitstream = ''.join(f'{byte:08b}' for byte in all_data)
    
    components = {}
    bit_pos = 0
    
    for name in ['Y', 'Cb', 'Cr']:
        dc_length = lengths[name]['dc_length']
        if bit_pos + dc_length > len(full_bitstream):
            raise ValueError(f"Недостаточно данных для {name} DC (нужно {dc_length}, доступно {len(full_bitstream) - bit_pos})")
        
        components[name] = {
            'dc': full_bitstream[bit_pos:bit_pos + dc_length]
        }
        bit_pos += dc_length
        
        if dc_length % 8 != 0:
            align_bits = 8 - (dc_length % 8)
            bit_pos += align_bits
        
        ac_length = lengths[name]['ac_length']
        components[name]['ac'] = full_bitstream[bit_pos:bit_pos + ac_length]
        bit_pos += ac_length
        
        if ac_length % 8 != 0:
            bit_pos += 8 - (ac_length % 8)

    return {
        'width': width,
        'height': height,
        'qf': qf,
        'lengths': lengths
    }, components

def decode_huffman_dc(bitstream, flag):
    huffman_table = DC_LUMA_HUFFMAN if flag == 0 else DC_CHROM_HUFFMAN
    
    code = ''
    for bit in bitstream:
        code += bit
        for key, value in huffman_table.items():
            if value == code:
                remaining_bits = bitstream[len(code):]
                return key, remaining_bits
    raise ValueError("Неверный DC Huffman код")

def decode_huffman_ac(bitstream, flag):
    huffman_table = AC_LUMA_HUFFMAN if flag == 0 else AC_CHROM_HUFFMAN
    reverse_table = {v: k for k, v in huffman_table.items()}
    
    code = ''
    max_code_length = max(len(code) for code in reverse_table.keys())
    
    for i, bit in enumerate(bitstream[:max_code_length + 16]):
        code += bit
        if code in reverse_table:
            key = reverse_table[code]
            remaining_bits = bitstream[len(code):]
            
            if key == (0, 0):
                return (0, 0, 0), remaining_bits
            elif key == (15, 0):
                return (15, 0, 0), remaining_bits
                
            run, size = key
            
            if len(remaining_bits) < size:
                raise ValueError(f"Недостаточно битов для значения. Нужно {size}, доступно {len(remaining_bits)}")
            
            value_bits = remaining_bits[:size]
            remaining_bits = remaining_bits[size:]

            
            return (run, size, value_bits), remaining_bits
    
    print(f"Ошибка декодирования AC коэффициента:")
    print(f"Флаг: {'Luma' if flag == 0 else 'Chroma'}")
    print(f"Начало битового потока: {bitstream[:32]}")
    print(f"Проверенные коды: {code}")
    print(f"Доступные коды в таблице: {list(reverse_table.keys())[:10]}...")
    
    raise ValueError(f"Неверный AC Huffman код. Не найдено совпадение для префикса {code} в первых {max_code_length + 16} битах")

def decode_dc_coefficients(bitstream, num_blocks, flag, max_bits=None):
    dc_coeffs = []
    prev_dc = 0
    remaining_bits = bitstream
    
    for _ in range(num_blocks):
        if max_bits is not None and len(remaining_bits) > max_bits:
            raise ValueError("Превышена ожидаемая длина битового потока DC")
            
        if not remaining_bits:
            dc_coeffs.append(prev_dc)
            continue
            
        try:
            size, new_remaining = decode_huffman_dc(remaining_bits, flag)
        except ValueError:
            dc_coeffs.append(prev_dc)
            continue
            
        if size > 0:
            if len(new_remaining) < size:
                dc_coeffs.append(prev_dc)
                remaining_bits = ''
                continue
                
            value_bits = new_remaining[:size]
            remaining_bits = new_remaining[size:]
            
            if value_bits[0] == '1':
                value = int(value_bits, 2)
            else: 
                value = int(value_bits, 2) - (1 << size) + 1
        else:
            value = 0
            remaining_bits = new_remaining
            
        dc_value = prev_dc + value
        dc_coeffs.append(dc_value)
        prev_dc = dc_value
    
    return dc_coeffs, remaining_bits

def reverse_zigzag(coefficients, block_size=8):
    block = np.zeros((block_size, block_size), dtype=np.int16)
    n = block_size
    
    zigzag_order = []
    for s in range(0, 2*n-1):
        if s < n:
            if s % 2 == 0:
                i, j = s, 0
                while i >= 0 and j < n:
                    zigzag_order.append((i, j))
                    i -= 1
                    j += 1
            else:
                i, j = 0, s
                while j >= 0 and i < n:
                    zigzag_order.append((i, j))
                    i += 1
                    j -= 1
        else:
            if s % 2 == 0:
                i, j = n-1, s-n+1
                while i >= 0 and j < n:
                    zigzag_order.append((i, j))
                    i -= 1
                    j += 1
            else:
                i, j = s-n+1, n-1
                while j >= 0 and i < n:
                    zigzag_order.append((i, j))
                    i += 1
                    j -= 1
    
    for idx, (i, j) in enumerate(zigzag_order):
        if idx < len(coefficients):
            block[i, j] = coefficients[idx]
        else:
            block[i, j] = 0 
    
    return block


def variable_encoding(x):
    x_int = int(round(x))
    if x_int == 0:
        return (0, '')
    
    l = 0
    while abs(x_int) >= (1 << l):
        l += 1
    
    if x_int > 0:
        bits = bin(x_int)[2:].zfill(l)
    else:
        bits = bin(x_int + (1 << l) - 1)[2:].zfill(l)
    
    return (l, bits[-l:] if l > 0 else '')

def encode_ac_coefficients(ac_coeffs, flag):
    encoded_bits = []
    
    for block_idx, block_ac in enumerate(ac_coeffs):
        if all(coeff == 0 for coeff in block_ac):
            encoded_bits.append(encode_ac_with_huffman(0, 0, 0, flag))
            continue
            
        var_encoded = []
        non_zero_count = 0
        for coeff in block_ac:
            size, bits = variable_encoding(coeff)
            var_encoded.append((size, bits))
            if coeff != 0:
                non_zero_count += 1
        if block_idx == 81:
            print(var_encoded, len(var_encoded))
        rle_encoded = []
        zero_run = 0
        coeff_count = 0
        
        for size, bits in var_encoded:
            if size == 0:
                zero_run += 1
                if zero_run == 16:
                    rle_encoded.append((15, 0, 0))
                    zero_run = 0
            else:
                if zero_run > 0:
                    rle_encoded.append((zero_run, size, int(bits, 2)))
                    zero_run = 0
                else:
                    rle_encoded.append((0, size, int(bits, 2)))
        if len(rle_encoded) < 63 and rle_encoded[-1] != (0, 0, 0):
            rle_encoded.append((0, 0, 0))
        else:
            print(f"Блок {block_idx} заполнен, EOB пропущен", len(rle_encoded))
        

        if len(rle_encoded) == 1 and rle_encoded[0] == (0, 0, 0):
            print(f"Ошибка: Блок {block_idx} закодирован как только EOB, но содержит ненулевые коэффициенты!")
            print("Ненулевые коэффициенты:", [coeff for coeff in block_ac if coeff != 0])
        
        for run, size, value in rle_encoded:
            encoded_bits.append(encode_ac_with_huffman(run, size, value, flag))

    return ''.join(encoded_bits)

def decode_ac_coefficients(bitstream, num_blocks, flag, max_bits=None):
    all_ac_coeffs = []
    remaining_bits = bitstream
    
    for _ in range(num_blocks):
        block_ac = []
        eob_found = False
        coeff_count = 0
        
        while not eob_found and coeff_count < 63 and remaining_bits:
            try:
                (run, size, value_bits), remaining_bits = decode_huffman_ac(remaining_bits, flag)
                
                if (run, size) == (0, 0):
                    eob_found = True
                    break
                elif (run, size) == (15, 0):
                    block_ac.extend([0] * 16)
                    coeff_count += 16
                    if (_==81):
                        print(coeff_count)
                else:
                    if run > 0:
                        block_ac.extend([0] * run)
                        coeff_count += run
                        if (_==81):
                            print(coeff_count)
                    
                    if size > 0:
                        if not value_bits:
                            value = 0
                        else:
                            if value_bits[0] == '0':  
                                value = int(value_bits, 2) - (1 << size) + 1
                            else:  
                                value = int(value_bits, 2)
                        block_ac.append(value)
                        coeff_count += 1
                        if (_==81):
                            print(coeff_count)
                    else:
                        block_ac.append(0)
                        coeff_count += 1
                        if (_==81):
                            print(coeff_count)
            except ValueError as e:
                print(f"AC decoding error in block {_}: {e}")
                break
        
        if coeff_count == 63 and eob_found == False:
            (run, size, value_bits), remaining_bits = decode_huffman_ac(remaining_bits, flag)
        if coeff_count < 63:
            block_ac.extend([0] * (63 - coeff_count))

        all_ac_coeffs.append(block_ac)
    
    return all_ac_coeffs, remaining_bits

def decode_value(value_bits, size):
    if not value_bits:
        return 0
    value = int(value_bits, 2)
    if value >= (1 << (size - 1)):
        return value
    else:
        return value - (1 << size) + 1

def write_jpeg_raw(components, output_path):
    with open(output_path, 'wb') as f:
        f.write(components['Y']['weight'].to_bytes(2, 'big'))
        f.write(components['Y']['height'].to_bytes(2, 'big'))
        f.write(components['Y']['qf'].to_bytes(1, 'big'))
        
        for name in ['Y', 'Cb', 'Cr']:
            comp = components[name]

            f.write(comp['weight'].to_bytes(2, 'big'))
            f.write(comp['height'].to_bytes(2, 'big'))

            f.write(comp['quant_matrix'].astype(np.uint8).tobytes())

            dc_encoded = comp['dc'].encode('utf-8')
            f.write(len(dc_encoded).to_bytes(4, 'big'))
            f.write(dc_encoded)

            ac_encoded = comp['ac'].encode('utf-8')
            f.write(len(ac_encoded).to_bytes(4, 'big'))
            f.write(ac_encoded)

def read_jpeg_raw(input_path):
    components = {}
    
    with open(input_path, 'rb') as f:
        y_width = int.from_bytes(f.read(2), 'big')
        y_height = int.from_bytes(f.read(2), 'big')
        qf = int.from_bytes(f.read(1), 'big')
        
        for name in ['Y', 'Cb', 'Cr']:
            comp_width = int.from_bytes(f.read(2), 'big')
            comp_height = int.from_bytes(f.read(2), 'big')
            
            quant_matrix = np.frombuffer(f.read(64), dtype=np.uint8).reshape(8, 8)
            
            dc_len = int.from_bytes(f.read(4), 'big')
            dc_data = f.read(dc_len).decode('utf-8')
            
            ac_len = int.from_bytes(f.read(4), 'big')
            ac_data = f.read(ac_len).decode('utf-8')
            
            components[name] = {
                'weight': comp_width,
                'height': comp_height,
                'qf': qf,
                'quant_matrix': quant_matrix,
                'dc': dc_data,
                'ac': ac_data
            }
    
    return components

def jpeg_compress_no_quant(input_path, qf=90, N=512):
    y, cb, cr = raw_to_ycbcr(input_path, N, N)
    
    y_c = y 
    cb_c = downsample_image(cb) 
    cr_c = downsample_image(cr)  
    
    components = {}
    quant_blocks_t = {}
    matrix = {}
    for name, component in zip(['Y', 'Cb', 'Cr'], [y_c, cb_c, cr_c]):
        block_size = 8
        comp_h, comp_w = component.shape

        if name=='Y':
            quant_matrix = get_quantization_matrix(qf, True)
            flag=0
        else:
            quant_matrix = get_quantization_matrix(qf, False)
            flag=1
        blocks = splitting_img(block_size, component)
        dct_blocks = [DCT(block) for block in blocks]
        quant_blocks = [quantize(block, quant_matrix) for block in dct_blocks]
        print("1")
        quant_blocks_t[name]=quant_blocks
        matrix[name]=quant_matrix
        dc_coeffs = [block[0,0] for block in quant_blocks]
        delta_dc = encode_dc(dc_coeffs)
        dc_bits = [encode_dc_with_huffman(delta, flag) for delta in delta_dc]
        dc_bitstream = ''.join(dc_bits)
        
        ac_coeffs = []
        for block in quant_blocks:
            zz = zigzag(block)
            ac_coeffs.append(zz[1:])  
        
        ac_bitstream = encode_ac_coefficients(ac_coeffs, flag)

        components[name] = {
            'weight': component.shape[1],
            'height': component.shape[0],
            'qf': qf,
            'quant_matrix': quant_matrix,
            'dc': dc_bitstream,
            'ac': ac_bitstream,
        }
    return components

def jpeg_decompress_no_quant(components, output_path, quant_blocks_true , quant_matrix):
    restored = {}
    for name in ['Y', 'Cb', 'Cr']:
        weidth = components[name]['weight']
        height = components[name]['height']
        quant_matrix = components[name]['quant_matrix']

        flag = 0 if name == 'Y' else 1
        
        num_blocks = (weidth // 8) ** 2

        dc_coeffs, _ = decode_dc_coefficients(components[name]['dc'], num_blocks, flag)
        print("Успешное dc декодирование")
        
        ac_coeffs, _ = decode_ac_coefficients(components[name]['ac'], num_blocks, flag)
        print("Успешное ac декодирование")

        quant_blocks = []
        for i in range(num_blocks):
            all_coeffs = [dc_coeffs[i]] + ac_coeffs[i]
            block = np.zeros((8, 8), dtype=np.int16)
            zigzag_indices = [
                (0,0), (0,1), (1,0), (2,0), (1,1), (0,2), (0,3), (1,2),
                (2,1), (3,0), (4,0), (3,1), (2,2), (1,3), (0,4), (0,5),
                (1,4), (2,3), (3,2), (4,1), (5,0), (6,0), (5,1), (4,2),
                (3,3), (2,4), (1,5), (0,6), (0,7), (1,6), (2,5), (3,4),
                (4,3), (5,2), (6,1), (7,0), (7,1), (6,2), (5,3), (4,4),
                (3,5), (2,6), (1,7), (2,7), (3,6), (4,5), (5,4), (6,3),
                (7,2), (7,3), (6,4), (5,5), (4,6), (3,7), (4,7), (5,6),
                (6,5), (7,4), (7,5), (6,6), (5,7), (6,7), (7,6), (7,7)
            ]
            for idx, (y, x) in enumerate(zigzag_indices):
                if idx < len(all_coeffs):
                    block[y, x] = all_coeffs[idx]
                else:
                    block[y, x] = 0 
    
            quant_blocks.append(block)
        weidth, height = 2048, 2048
        quant_blocks = quant_blocks_true[name]
        dct_blocks = [dequantize(block, quant_matrix[name]) for block in quant_blocks]
        blocks = [iDCT(block) for block in dct_blocks]

        comp = np.zeros((weidth,height), dtype=np.float32)
        blocks_per_row = (weidth + 7) // 8 
        
        for i, block in enumerate(blocks):
            row = (i // blocks_per_row) * 8
            col = (i % blocks_per_row) * 8
            
            row_end = min(row + 8, height)
            col_end = min(col + 8, height)
            
            block_height = row_end - row
            block_width = col_end - col
            actual_block = block[:block_height, :block_width]
            
            comp[row:row_end, col:col_end] = actual_block
        
        restored[name] = np.clip(comp, 0, 255).astype(np.uint8)
        print('2')
    
    cb_up = upsample_image(restored['Cb'], scale_factor=2)
    cr_up = upsample_image(restored['Cr'], scale_factor=2)
    
    y = restored['Y'].astype(np.float32)
    cb = cb_up.astype(np.float32) - 128
    cr = cr_up.astype(np.float32) - 128
    
    r = np.clip(y + 1.402 * cr, 0, 255)
    g = np.clip(y - 0.34414 * cb - 0.71414 * cr, 0, 255)
    b = np.clip(y + 1.772 * cb, 0, 255)
    
    rgb = np.stack([r, g, b], axis=2).astype(np.uint8)
    Image.fromarray(rgb).save(output_path)


import matplotlib.pyplot as plt
import os

def plot_compression_ratio_no_quant(input_path, output_dir, N=512):
    qf_values = range(5, 101, 5)
    file_sizes = []
    
    os.makedirs(output_dir, exist_ok=True)
    
    for qf in qf_values:
        temp_file = os.path.join(output_dir, f"temp_{qf}.raw")
        components = jpeg_compress_no_quant(input_path, qf, N=N)
        write_jpeg_raw(components, temp_file)
        
        file_size = os.path.getsize(temp_file) / 1024
        file_sizes.append(file_size)
        
        print(f"QF={qf}: {file_size:.2f} KB")
        os.remove(temp_file)
    
    plt.figure(figsize=(10, 6))
    plt.plot(qf_values, file_sizes, 'ro-', linewidth=2, markersize=8)
    plt.title('Зависимость размера сжатого файла от коэффициента качества')
    plt.xlabel('Коэффициент качества (qf)')
    plt.ylabel('Размер файла (KB)')
    plt.grid(True)
    plt.xticks(range(0, 101, 10))
    
    plot_path = os.path.join(output_dir, 'compression_ratio_no_quant.png')
    plt.savefig(plot_path)
    plt.close()
    
    print(f"График сохранен в {plot_path}")

# if __name__ == "__main__":
#     input_file = "Lenna.raw"
#     output_directory = "compression_results_no_quant"
    
#     plot_compression_ratio_no_quant(input_file, output_directory, N=512)


def png_to_raw(input_path, output_path):
    img = Image.open(input_path)
    img_array = np.array(img)
    with open(output_path, 'wb') as f:
        f.write(img_array.tobytes())
    print(f"Сохранено RAW изображение: {output_path}")
    print(f"Размеры: {img_array.shape}, Тип данных: {img_array.dtype}")

def raw_to_png(input_path, output_path, width, height, channels=1, dtype=np.uint8):
    with open(input_path, 'rb') as f:
        raw_data = f.read()
    
    img_array = np.frombuffer(raw_data, dtype=dtype)
    img_array = img_array.reshape((height, width)) if channels == 1 else img_array.reshape((height, width, channels))
    
    img = Image.fromarray(img_array)
    img.save(output_path)
    print(f"Сохранено PNG изображение: {output_path}")

def load_bw_image(input_path, width, height):
    """Загрузка ЧБ изображения с сохранением оригинальных значений"""
    with open(input_path, 'rb') as f:
        raw_data = f.read()
    img_array = np.frombuffer(raw_data, dtype=np.uint8).reshape((height, width))
    return Image.fromarray(img_array)

def load_raw_image(input_path, width, height):
    with open(input_path, 'rb') as f:
        raw_data = f.read()
    img_array = np.frombuffer(raw_data, dtype=np.uint8).reshape((height, width))
    return Image.fromarray(img_array)

def DCT_g(A):
    N = A.shape[0]
    coeffs, scale = precompute_dct_coeffs(N)
    B = np.zeros((N, N))
    
    if A.dtype == np.uint8:
        A = A.astype(np.float32) - 128
    
    for p in range(N):
        for q in range(N):
            total = 0.0
            for m in range(N):
                for n in range(N):
                    total += A[m,n] * coeffs[p,q,m,n]
            B[p,q] = total * scale[p,q] * (2/N)
    
    return B.round().astype(np.int16)

def iDCT_g(B):
    N = B.shape[0]
    coeffs, scale = precompute_dct_coeffs(N)
    A = np.zeros((N, N))
    
    for m in range(N):
        for n in range(N):
            total = 0.0
            for p in range(N):
                for q in range(N):
                    total += B[p,q] * coeffs[p,q,m,n] * scale[p,q]
            A[m,n] = total * (2/N)
    
    A = np.clip(A + 128, 0, 255).round().astype(np.uint8)
    return A

def jpeg_compress_monochrome(input_path, output_path, qf=90, N=512, is_bw=False):
    # img = load_raw_image(input_path, N, N)
    
    if is_bw:
        img = load_bw_image(input_path, N, N) 
    else:
        img = load_raw_image(input_path, N, N)
    
    img_array = np.array(img)
    
    img_array = img_array[:N, :N]
    
    if is_bw:
        img_array = np.where(img_array > 128, 255, 0).astype(np.uint8)

    blocks = splitting_img(8, img_array)
    

    # dct_blocks = np.array([DCT(block) for block in blocks])
    dct_blocks = np.array([DCT_g(block) for block in blocks])
    quant_blocks = np.array([quantize(block, get_quantization_matrix(qf)) for block in dct_blocks])
    
    dc_coeffs = [block[0,0] for block in quant_blocks]
    delta_dc = encode_dc(dc_coeffs)
    dc_bits = [encode_dc_with_huffman(delta, 0) for delta in delta_dc] 
    dc_bitstream = ''.join(dc_bits)
    
    ac_coeffs = []
    for block in quant_blocks:
        zz = zigzag(block)
        ac_coeffs.append(zz[1:])
        
    ac_bitstream = encode_ac_coefficients(ac_coeffs, 0)
    
    metadata = {
        'width': N,
        'height': N,
        'qf': qf,
        'is_bw': is_bw,
        'dc_length': len(dc_bitstream),
        'ac_length': len(ac_bitstream)
    }
    
    with open(output_path, 'wb') as f:
        f.write(metadata['width'].to_bytes(4, byteorder='big'))
        f.write(metadata['height'].to_bytes(4, byteorder='big'))
        f.write(metadata['qf'].to_bytes(1, byteorder='big'))
        f.write(int(metadata['is_bw']).to_bytes(1, byteorder='big'))
        
        f.write(metadata['dc_length'].to_bytes(4, byteorder='big'))
        f.write(metadata['ac_length'].to_bytes(4, byteorder='big'))
        
        dc_padded = dc_bitstream + '0' * ((8 - len(dc_bitstream) % 8) % 8)
        f.write(bytes(int(dc_padded[i:i+8], 2) for i in range(0, len(dc_padded), 8)))
        
        ac_padded = ac_bitstream + '0' * ((8 - len(ac_bitstream) % 8) % 8)
        f.write(bytes(int(ac_padded[i:i+8], 2) for i in range(0, len(ac_padded), 8)))
    
def jpeg_decompress_monochrome(input_path, output_path):
    with open(input_path, 'rb') as f:
        width = int.from_bytes(f.read(4), byteorder='big')
        height = int.from_bytes(f.read(4), byteorder='big')
        qf = int.from_bytes(f.read(1), byteorder='big')
        is_bw = bool(int.from_bytes(f.read(1), byteorder='big'))
        
        dc_length = int.from_bytes(f.read(4), byteorder='big')
        ac_length = int.from_bytes(f.read(4), byteorder='big')
        
        dc_bytes = f.read((dc_length + 7) // 8)
        dc_bitstream = ''.join(f'{byte:08b}' for byte in dc_bytes)[:dc_length]
        
        ac_bytes = f.read((ac_length + 7) // 8)
        ac_bitstream = ''.join(f'{byte:08b}' for byte in ac_bytes)[:ac_length]
    
    num_blocks = ((width + 7) // 8) * ((height + 7) // 8)
    dc_coeffs, _ = decode_dc_coefficients(dc_bitstream, num_blocks, 0)
    ac_coeffs, _ = decode_ac_coefficients(ac_bitstream, num_blocks, 0)
    
    quant_blocks = []
    for i in range(num_blocks):
        all_coeffs = [dc_coeffs[i]] + ac_coeffs[i]
        block = reverse_zigzag(all_coeffs)
        quant_blocks.append(block)
    Q = get_quantization_matrix(qf)
    dct_blocks = [dequantize(block, Q) for block in quant_blocks]
    # blocks = [iDCT(block) for block in dct_blocks]
    blocks = [iDCT_g(block) for block in dct_blocks]
    img_array = np.zeros((height, width), dtype=np.uint8)
    blocks_per_row = (width + 7) // 8
    
    for i, block in enumerate(blocks):
        row = (i // blocks_per_row) * 8
        col = (i % blocks_per_row) * 8
        
        h = min(8, height - row)
        w = min(8, width - col)
        
        img_array[row:row+h, col:col+w] = block[:h, :w]
    
    if is_bw:
        img_array = np.where(img_array > 128, 255, 0).astype(np.uint8)
        img = Image.fromarray(img_array).convert('1')  
    else:
        img = Image.fromarray(img_array)
    
    img = Image.fromarray(img_array)
    img.save(output_path)
    print(f"Изображение успешно восстановлено: {output_path}")

def jpeg_compress_monochrome_bw(input_path, output_path, qf=90, N=512, is_bw=False):
    if is_bw:
        img = load_bw_image(input_path, N, N)
       
        img_array = np.array(img)
        img_array = np.where(img_array > 0, 255, 0).astype(np.uint8)
    else:
        img = load_raw_image(input_path, N, N)
        img_array = np.array(img)
    
    img_array = img_array[:N, :N]
    
    blocks = splitting_img(8, img_array)
    
    dct_blocks = np.array([DCT_g(block) for block in blocks])
    quant_blocks = np.array([quantize(block, get_quantization_matrix(qf)) for block in dct_blocks])
    
    dc_coeffs = [block[0,0] for block in quant_blocks]
    delta_dc = encode_dc(dc_coeffs)
    dc_bits = [encode_dc_with_huffman(delta, 0) for delta in delta_dc]
    dc_bitstream = ''.join(dc_bits)
    
    ac_coeffs = []
    for block in quant_blocks:
        zz = zigzag(block)
        ac_coeffs.append(zz[1:])
        
    ac_bitstream = encode_ac_coefficients(ac_coeffs, 0)
    
    metadata = {
        'width': N,
        'height': N,
        'qf': qf,
        'is_bw': is_bw,
        'dc_length': len(dc_bitstream),
        'ac_length': len(ac_bitstream)
    }
    
    with open(output_path, 'wb') as f:
        f.write(metadata['width'].to_bytes(4, byteorder='big'))
        f.write(metadata['height'].to_bytes(4, byteorder='big'))
        f.write(metadata['qf'].to_bytes(1, byteorder='big'))
        f.write(int(metadata['is_bw']).to_bytes(1, byteorder='big'))
        
        f.write(metadata['dc_length'].to_bytes(4, byteorder='big'))
        f.write(metadata['ac_length'].to_bytes(4, byteorder='big'))
        
        dc_padded = dc_bitstream + '0' * ((8 - len(dc_bitstream) % 8) % 8)
        f.write(bytes(int(dc_padded[i:i+8], 2) for i in range(0, len(dc_padded), 8)))
        
        ac_padded = ac_bitstream + '0' * ((8 - len(ac_bitstream) % 8) % 8)
        f.write(bytes(int(ac_padded[i:i+8], 2) for i in range(0, len(ac_padded), 8)))
    return dc_coeffs, ac_coeffs

def jpeg_decompress_monochrome_bw(input_path, output_path, dc, ac):
    with open(input_path, 'rb') as f:
        width = int.from_bytes(f.read(4), byteorder='big')
        height = int.from_bytes(f.read(4), byteorder='big')
        qf = int.from_bytes(f.read(1), byteorder='big')
        is_bw = bool(int.from_bytes(f.read(1), byteorder='big'))
        
        dc_length = int.from_bytes(f.read(4), byteorder='big')
        ac_length = int.from_bytes(f.read(4), byteorder='big')
        
        dc_bytes = f.read((dc_length + 7) // 8)
        dc_bitstream = ''.join(f'{byte:08b}' for byte in dc_bytes)[:dc_length]
        
        ac_bytes = f.read((ac_length + 7) // 8)
        ac_bitstream = ''.join(f'{byte:08b}' for byte in ac_bytes)[:ac_length]
    
    num_blocks = ((width + 7) // 8) * ((height + 7) // 8)
    dc_coeffs, _ = decode_dc_coefficients(dc_bitstream, num_blocks, 0)
    ac_coeffs, _ = decode_ac_coefficients(ac_bitstream, num_blocks, 0)
    # dc_coeffs = dc
    ac_coeffs = ac
    quant_blocks = []
    for i in range(num_blocks):
        all_coeffs = [dc_coeffs[i]] + ac_coeffs[i]
        block = reverse_zigzag(all_coeffs)
        quant_blocks.append(block)

    Q = get_quantization_matrix(qf)
    dct_blocks = [dequantize(block, Q) for block in quant_blocks]
    # blocks = [iDCT(block) for block in dct_blocks]
    blocks = [iDCT_g(block) for block in dct_blocks]
    img_array = np.zeros((height, width), dtype=np.uint8)
    blocks_per_row = (width + 7) // 8
    
    for i, block in enumerate(blocks):
        row = (i // blocks_per_row) * 8
        col = (i % blocks_per_row) * 8
        
        h = min(8, height - row)
        w = min(8, width - col)
        
        img_array[row:row+h, col:col+w] = block[:h, :w]
    
    if is_bw:
        img_array = np.where(img_array > 128, 255, 0).astype(np.uint8)
        img = Image.fromarray(img_array).convert('1')
    else:
        img = Image.fromarray(img_array)
    
    img.save(output_path)
    print(f"Изображение успешно восстановлено: {output_path}")


def write_jpeg_raw_mono(components, output_path):
    with open(output_path, 'wb') as f:
        width = components['width']
        height = components['height']
        qf = components['qf']
        is_bw = components['is_bw']
        
        f.write(width.to_bytes(4, byteorder='big'))
        f.write(height.to_bytes(4, byteorder='big'))
        f.write(qf.to_bytes(1, byteorder='big'))
        f.write(int(is_bw).to_bytes(1, byteorder='big'))
        
        dc_bitstream = components['Y']['dc']
        ac_bitstream = components['Y']['ac']
        
        f.write(len(dc_bitstream).to_bytes(4, byteorder='big'))
        f.write(len(ac_bitstream).to_bytes(4, byteorder='big'))
        
        dc_padded = dc_bitstream + '0' * ((8 - len(dc_bitstream) % 8) % 8)
        f.write(bytes(int(dc_padded[i:i+8], 2) for i in range(0, len(dc_padded), 8)))
        
        ac_padded = ac_bitstream + '0' * ((8 - len(ac_bitstream) % 8) % 8)
        f.write(bytes(int(ac_padded[i:i+8], 2) for i in range(0, len(ac_padded), 8)))

def plot_compression_ratio_no_quant(input_path, output_dir, N=512):
    qf_values = range(5, 101, 5)
    file_sizes = []
    
    os.makedirs(output_dir, exist_ok=True)
    
    # for qf in qf_values:
    #     # Генерируем имя выходного файла
    #     output_path = os.path.join(output_dir, f"Lighthouse_qf_{qf}.jgc")
        
    #     # Сжимаем изображение
    #     # jpeg_compress_monochrome(input_path, output_path, qf=qf, N=N, is_bw=False)
    #     jpeg_compress_monochrome_bw(input_path, output_path, qf=qf, N=N, is_bw=True)
    #     # Получаем размер файла в КБ
    #     file_size = os.path.getsize(output_path) / 1024
    #     file_sizes.append(file_size)
        
    #     print(f"QF={qf}: {file_size:.2f} KB")
    for qf in qf_values:
        temp_file = os.path.join(output_dir, f"temp_{qf}.raw")
        components = jpeg_compress_no_quant(input_path, qf, N=N)
        write_jpeg_raw(components, temp_file)
        
        file_size = os.path.getsize(temp_file) / 1024
        file_sizes.append(file_size)
        
        print(f"QF={qf}: {file_size:.2f} KB")
        os.remove(temp_file)
    
    plt.figure(figsize=(10, 6))
    plt.plot(qf_values, file_sizes, 'bo-', linewidth=2, markersize=8)
    
    title = 'Зависимость размера сжатого файла от коэффициента качества (Цветная)'

    
    plt.title(title)
    plt.xlabel('Коэффициент качества (qf)')
    plt.ylabel('Размер файла (KB)')
    plt.grid(True)
    plt.xticks(range(0, 101, 10))
    
    plot_name = 'Lighthouse_graf.png'
    plot_path = os.path.join(output_dir, plot_name)
    plt.savefig(plot_path)
    plt.close()
    
    print(f"График сохранен в {plot_path}")

if __name__ == "__main__":
    input_file = "Lighthouse.raw"
    output_directory = "compression_results_no_quant"
    plot_compression_ratio_no_quant(input_file, output_directory, 2048)

def jpeg_compress_and_decompress(input_path, output_path=None, qf=90, N=512, is_bw=False):
    y, cb, cr = raw_to_ycbcr(input_path, N, N)
    original_size = y.nbytes + cb.nbytes + cr.nbytes

    cb_down = downsample_image(cb)
    cr_down = downsample_image(cr)
    
    components = {
        'Y': {'data': y, 'quant_matrix': get_quantization_matrix(qf, is_luma=True)},
        'Cb': {'data': cb_down, 'quant_matrix': get_quantization_matrix(qf, is_luma=False)},
        'Cr': {'data': cr_down, 'quant_matrix': get_quantization_matrix(qf, is_luma=False)}
    }

    compressed_blocks = {}
    for name in ['Y', 'Cb', 'Cr']:
        comp = components[name]
        blocks = splitting_img(8, comp['data'])
        dct_blocks = [DCT_g(block) for block in blocks]
        quant_blocks = [quantize(block, comp['quant_matrix']) for block in dct_blocks]
        compressed_blocks[name] = quant_blocks
    print("123")
    restored = {}
    for name in ['Y', 'Cb', 'Cr']:
        quant_blocks = compressed_blocks[name]
        dct_blocks = [dequantize(block, components[name]['quant_matrix']) for block in quant_blocks]
        blocks = [iDCT_g(block) for block in dct_blocks]

        comp_data = np.zeros_like(components[name]['data'])
        blocks_per_row = (comp_data.shape[1] + 7) // 8
        
        for i, block in enumerate(blocks):
            row = (i // blocks_per_row) * 8
            col = (i % blocks_per_row) * 8
            h, w = min(8, comp_data.shape[0]-row), min(8, comp_data.shape[1]-col)
            comp_data[row:row+h, col:col+w] = block[:h, :w]
        
        if name in ['Cb', 'Cr']:
            comp_data = upsample_image(comp_data)
        
        restored[name] = np.clip(comp_data, 0, 255).astype(np.uint8)

    y = restored['Y'].astype(np.float32)
    cb = restored['Cb'].astype(np.float32) - 128
    cr = restored['Cr'].astype(np.float32) - 128
    
    r = np.clip(y + 1.402 * cr, 0, 255)
    g = np.clip(y - 0.34414 * cb - 0.71414 * cr, 0, 255)
    b = np.clip(y + 1.772 * cb, 0, 255)
    
    rgb_image = np.stack([r, g, b], axis=2).astype(np.uint8)

    if output_path:
        Image.fromarray(rgb_image).save(output_path)
    
    compressed_size = sum(
        block.nbytes for name in compressed_blocks 
        for block in compressed_blocks[name]
    )
    ratio = original_size / compressed_size

    return rgb_image, ratio

if __name__ == "__main__":
    restored_img, ratio = jpeg_compress_and_decompress(
        "Lighthouse.raw", "Lighthouse_restored_color.png", qf=90, N=2048)
    
    # restored_bw, ratio_bw = jpeg_compress_and_decompress(
    #     "Lenna_bw.raw", "restored_bw.jpg", qf=90, N=512, is_bw=True)
    # print(f"Коэффициент сжатия ЧБ: {ratio_bw:.2f}:1")
# if __name__ == "__main__":

    # for qf in range(0,120,20):    
    #     components = jpeg_compress_no_quant("Lighthouse.raw", qf, 2048)
    #     write_jpeg_raw(components, f"Lighthouse_Data_{qf}")
    #     print("Данные записаны")
    #     comp=read_jpeg_raw(f"Lighthouse_Data_{qf}")
    #     print("Данные прочитаны")
    #     jpeg_decompress_no_quant(comp, f"Lighthouse_restored_{qf}.png")
    # qf=90
    # comp = jpeg_compress_no_quant("Lighthouse.raw", qf, 2048)
    # write_jpeg_raw(components, f"Lighthouse_Data_{qf}")
    # print("Данные записаны")
    # comp=read_jpeg_raw(f"Lighthouse_Data_{qf}")
    # print("Данные прочитаны")
    # jpeg_decompress_no_quant(comp, f"Lighthouse_restored_{qf}.png")

    # for qf in range(0,120,20):
        
    #     # jpeg_compress_monochrome("Lenna_grayscale.raw", f"Lenna_grayscale_data_{qf}.raw", qf, N=512, is_bw=False)
    #     jpeg_compress_monochrome("Lenna_grayscale.raw", f"Lenna_grayscale_data_{qf}.raw", qf, N=512, is_bw=False)
    #     print("Компрессия")
    #     jpeg_decompress_monochrome(f"Lenna_grayscale_data_{qf}.raw", f"Lenna_grayscale_restored_{qf}.png")
    #     print("Декомпрессия")
    
     # for qf in range(0,120,20):
        
     #     dc, ac = jpeg_compress_monochrome_bw("Lenna_blackwhite_dither.raw", f"Lenna_blackwhite_dither_data_{qf}.raw", qf, N=512, is_bw=True)
     #     print("Компрессия")
     #     jpeg_decompress_monochrome_bw(f"Lenna_blackwhite_dither_data_{qf}.raw", f"Lenna_blackwhite_dither_restored_{qf}.png", dc, ac)
     #     print("Декомпрессия")


     # qf=90
     # # jpeg_compress_monochrome_bw("Lenna_blackwhite_dither.raw", f"Lenna_blackwhite_dither_data_{qf}.raw", qf, N=512, is_bw=True)
     # # print("Компрессия")
     # # jpeg_decompress_monochrome_bw(f"Lenna_blackwhite_dither_data_{qf}.raw", f"Lenna_blackwhite_dither_restored_{qf}.png")
     # # print("Декомпрессия")
     # dc, ac = jpeg_compress_monochrome_bw("Lenna_blackwhite_no_dither.raw", f"Lenna_blackwhite_no_dither_data_{qf}.raw", qf, N=512, is_bw=True)
     # print("Компрессия")
     # jpeg_decompress_monochrome_bw(f"Lenna_blackwhite_no_dither_data_{qf}.raw", f"Lenna_blackwhite_no_dither_restored_{qf}.png", dc, ac)
     # print("Декомпрессия")
