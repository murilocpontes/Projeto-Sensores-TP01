import socket
import numpy as np
import pandas as pd
import warnings
from scipy.fft import rfft
from scipy.signal import butter, filtfilt

warnings.filterwarnings("ignore")

def butter_lowpass(data, cutoff, fs=50.0, order=3):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)

def extract_advanced_features(window_128x6):
    acc_total = window_128x6[:, 0:3]
    gyro = window_128x6[:, 3:6]

    acc_grav = butter_lowpass(acc_total, cutoff=0.3)
    acc_body = acc_total - acc_grav

    acc_body_jerk = np.diff(acc_body, axis=0, prepend=acc_body[0:1, :])
    gyro_jerk = np.diff(gyro, axis=0, prepend=gyro[0:1, :])
    
    acc_body_mag = np.linalg.norm(acc_body, axis=1, keepdims=True)
    acc_grav_mag = np.linalg.norm(acc_grav, axis=1, keepdims=True)
    gyro_mag = np.linalg.norm(gyro, axis=1, keepdims=True)

    acc_body_fft = np.abs(rfft(acc_body, axis=0))
    gyro_fft = np.abs(rfft(gyro, axis=0))

    sinais_tempo = [acc_body, acc_grav, gyro, acc_body_jerk, gyro_jerk, acc_body_mag, acc_grav_mag, gyro_mag]
    sinais_freq = [acc_body_fft, gyro_fft]

    features = []
    for sinal in sinais_tempo + sinais_freq:
        features.extend(np.mean(sinal, axis=0))
        features.extend(np.std(sinal, axis=0))
        features.extend(np.max(sinal, axis=0))
        features.extend(np.min(sinal, axis=0))
        features.extend(np.sum(sinal**2, axis=0) / sinal.shape[0])

    def get_correlations(matriz):
        cor_xy = np.corrcoef(matriz[:,0], matriz[:,1])[0,1]
        cor_xz = np.corrcoef(matriz[:,0], matriz[:,2])[0,1]
        cor_yz = np.corrcoef(matriz[:,1], matriz[:,2])[0,1]
        return [cor_xy, cor_xz, cor_yz]

    features.extend(np.nan_to_num(get_correlations(acc_body)))
    features.extend(np.nan_to_num(get_correlations(gyro)))

    return np.array(features, dtype=np.float32).reshape(1, -1)

UDP_IP = "0.0.0.0"
UDP_PORT = 5005
BUFFER_SIZE = 65535

# Mude isso aqui para a atividade que você vai fazer agora (ex: 1 para Walking)
CLASSE_ATUAL = 2  # 1: Walking, 2: Up, 3: Down, 4: Sitting, 5: Standing, 6: Laying
NOME_ARQUIVO = f"meus_dados_classe_{CLASSE_ATUAL}.csv"

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
print(f"[*] Gravando dados para a Classe {CLASSE_ATUAL}...")

features_coletadas = []

try:
    while True:
        data, addr = sock.recvfrom(BUFFER_SIZE)
        payload_str = data.decode("utf-8")
        linhas = payload_str.split(";")
        janela = [list(map(float, linha.split(","))) for linha in linhas if linha]
        window_numpy = np.array(janela)

        if window_numpy.shape == (128, 6):
            # Corrige a escala para 'g'
            window_numpy[:, 0:3] = window_numpy[:, 0:3] / 9.80665
            
            # Extrai o vetor de (~126) features da janela inteira
            vetor = extract_advanced_features(window_numpy)[0]
            features_coletadas.append(vetor)
            print(f"Janelas gravadas: {len(features_coletadas)} (Pressione Ctrl+C para parar)")

except KeyboardInterrupt:
    print("\n[*] Salvando arquivo CSV...")
    df = pd.DataFrame(features_coletadas)
    df['Label'] = CLASSE_ATUAL
    df.to_csv(NOME_ARQUIVO, index=False)
    print(f"[*] {len(features_coletadas)} janelas salvas em {NOME_ARQUIVO}!")