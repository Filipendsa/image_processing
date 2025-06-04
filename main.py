import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

# --- Funções Utilitárias ---

def read_images(image_path_pattern):
    paths = glob(image_path_pattern)
    data = []
    for path in paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255.0
        data.append(img)
    return data

def pad_image(image, pad=(8, 18)):
    return np.pad(image, pad, mode='edge')

def get_frequency(image, normalize=True, shift=True):
    f = np.fft.fft2(image)
    f_abs = np.abs(f)
    f_real = np.real(f)
    f_imag = np.imag(f)

    if normalize:
        f_abs = normalize_abs_freq(f_abs, shift)
        f_real = normalize_freq(f_real, shift)
        f_imag = normalize_freq(f_imag, shift)

    return f_abs, f_real, f_imag

def normalize_abs_freq(f_abs, shift):
    f_abs = np.log(1 + f_abs)
    f_abs -= np.min(f_abs)
    f_abs /= np.max(f_abs)
    if shift:
        f_abs = np.fft.fftshift(f_abs)
    return f_abs

def normalize_freq(freq, shift):
    freq -= np.min(freq)
    freq /= np.max(freq)
    if shift:
        freq = np.fft.fftshift(freq)
    return freq

def get_variance_map(freq_data, normalize=True):
    image_array = np.stack(freq_data, axis=-1)
    var_map = np.var(image_array, axis=2)
    return normalize_freq(var_map, shift=False) if normalize else var_map

def feature_distance(A, B):
    return np.sum(np.abs(A - B))

def display_freq(image, title_str):
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.title(title_str)
    plt.axis('off')
    plt.show()

# --- Parâmetros ---
db_path = 'att_faces/s*/'
image_path_pattern = 'att_faces/s*/*.pgm'
test_image = 'test_32_9.pgm'  # Altere conforme necessário

# --- Carregando a base ---
image_data = read_images(image_path_pattern)
padded_data = [pad_image(img) for img in image_data]

img_f_abs, img_f_real, img_f_imag = zip(*[get_frequency(img) for img in padded_data])
abs_var_map = get_variance_map(img_f_abs)
real_var_map = get_variance_map(img_f_real)
imag_var_map = get_variance_map(img_f_imag)

display_freq(abs_var_map, "Variance (Abs)")
display_freq(real_var_map, "Variance (Real)")
display_freq(imag_var_map, "Variance (Imag)")

# --- Imagem de teste ---
sample = cv2.imread(test_image, cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255.0
sample = pad_image(sample)
display_freq(sample, 'Sample Image')

sample_fft = np.fft.fft2(sample)
display_freq(normalize_abs_freq(np.abs(sample_fft), True), 'Sample FFT')

_, db_real, db_imag = zip(*[get_frequency(img, normalize=True, shift=False) for img in padded_data])
real_var_map = get_variance_map(db_real)
imag_var_map = get_variance_map(db_imag)

big_variance_real = real_var_map > 0.7
big_variance_imag = imag_var_map > 0.05

# --- Zerar frequências variáveis ---
sample_real = np.real(sample_fft).copy()
sample_imag = np.imag(sample_fft).copy()
sample_real[big_variance_real] = 0
sample_imag[big_variance_imag] = 0

sample_fft_zeroed = sample_real + 1j * sample_imag
recovered = np.fft.ifft2(sample_fft_zeroed)
display_freq(np.abs(recovered), "Recovered Image")

# --- Frequência chave ---
sample_real_key = np.real(sample_fft)
sample_imag_key = np.imag(sample_fft)
sample_real_key[~big_variance_real] = 0
sample_imag_key[~big_variance_imag] = 0

# --- Análise por pasta ---
subdirs = sorted([d for d in os.listdir("att_faces") if os.path.isdir(os.path.join("att_faces", d))])
frequency_real = []
frequency_imag = []

for sub in subdirs:
    imgs = read_images(os.path.join("att_faces", sub, "*.pgm"))
    padded_imgs = [pad_image(img) for img in imgs]
    _, real_parts, imag_parts = zip(*[get_frequency(img, normalize=False, shift=False) for img in padded_imgs])
    avg_real = np.mean(np.stack(real_parts, axis=-1), axis=2)
    avg_imag = np.mean(np.stack(imag_parts, axis=-1), axis=2)
    avg_real[~big_variance_real] = 0
    avg_imag[~big_variance_imag] = 0
    frequency_real.append(avg_real)
    frequency_imag.append(avg_imag)

# --- Comparação de distância ---
dist_real = [feature_distance(sample_real_key, fr) for fr in frequency_real]
dist_imag = [feature_distance(sample_imag_key, fi) for fi in frequency_imag]

idx_real = np.argmin(dist_real)
idx_imag = np.argmin(dist_imag)

print(f"Best match (real): s{idx_real + 1}")
print(f"Best match (imag): s{idx_imag + 1}")

# --- Mostra rosto previsto ---
i = np.random.randint(1, 11)
predicted_path = os.path.join("att_faces", f"s{idx_real+1}", f"{i}.pgm")
output = cv2.imread(predicted_path, cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255.0
output = pad_image(output)

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(sample, cmap='gray')
plt.title('Input')

plt.subplot(1, 2, 2)
plt.imshow(output, cmap='gray')
plt.title('Prediction')
plt.show()
