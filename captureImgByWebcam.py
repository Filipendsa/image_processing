import cv2
import os

# --- Parâmetros ---
subject_id = 41  # Altere para o número do seu sujeito
output_dir = f'att_faces/s{subject_id}'
os.makedirs(output_dir, exist_ok=True)

# --- Inicializa a câmera ---
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print("Pressione 'SPACE' para capturar imagem, 'ESC' para sair.")

img_count = 0
while img_count < 10:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (92, 112))  # Tamanho padrão da base

    cv2.imshow("Capture - Pressione SPACE", resized)

    key = cv2.waitKey(1)
    if key % 256 == 27:  # ESC
        print("Encerrado.")
        break
    elif key % 256 == 32:  # SPACE
        img_path = os.path.join(output_dir, f"{img_count + 1}.pgm")
        cv2.imwrite(img_path, resized)
        print(f"Imagem {img_count + 1} salva em {img_path}")
        img_count += 1

cap.release()
cv2.destroyAllWindows()
