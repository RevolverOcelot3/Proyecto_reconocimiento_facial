import cv2
import numpy as np
from tensorflow import keras
import json
import os

RUTA_MODELO = "reconocimientoentrenado.keras"
RUTA_CLASES = "clases.json"
TAMANO_IMAGEN = (160, 160)
UMBRAL_CONFIANZA = 0.90

ETIQUETAS_BONITAS = {
    "frame_yo": "Yo",
    "mamac": "Mama",
    "yahairac": "Yahaira",
    "ramsesc": "Ramses",
    "marsellac": "Marsella",
    "marianac": "Mariana",
    "javierc": "Javier",
    "jorgec": "Jorge",
    "cristoc": "Cristo",
    "antoc": "Anto",
}

if not os.path.exists(RUTA_MODELO):
    print(f"Error: No se encontro el archivo del modelo en '{RUTA_MODELO}'")
    exit()

if not os.path.exists(RUTA_CLASES):
    print(f"Error: No se encontro el archivo de clases en '{RUTA_CLASES}'")
    exit()

modelo = keras.models.load_model(RUTA_MODELO, compile=False)

with open(RUTA_CLASES, "r") as f:
    nombres_clases = json.load(f)

ruta_cascade = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
clasificador_caras = cv2.CascadeClassifier(ruta_cascade)

if clasificador_caras.empty():
    raise RuntimeError("No se pudo cargar el clasificador para rostros.")

def preprocesar_rostro(rostro_bgr: np.ndarray) -> np.ndarray:
    rostro_rgb = cv2.cvtColor(rostro_bgr, cv2.COLOR_BGR2RGB)
    rostro_redimensionado = cv2.resize(rostro_rgb, TAMANO_IMAGEN)
    rostro_array = rostro_redimensionado.astype("float32")
    rostro_listo = np.expand_dims(rostro_array, axis=0)
    return rostro_listo

def predecir_clase(rostro_bgr: np.ndarray):
    x = preprocesar_rostro(rostro_bgr)
    predicciones = modelo.predict(x, verbose=0)[0]
    indice_max = int(np.argmax(predicciones))
    probabilidad = float(predicciones[indice_max])
    
    if indice_max < len(nombres_clases):
        etiqueta = nombres_clases[indice_max]
    else:
        etiqueta = "Desconocido"

    return etiqueta, probabilidad

def etiqueta_bonita(etiqueta_raw: str) -> str:
    return ETIQUETAS_BONITAS.get(etiqueta_raw, etiqueta_raw)

def main():
    camara = cv2.VideoCapture(0)

    if not camara.isOpened():
        print("No se pudo abrir la camara.")
        return

    print("Camara abierta. Presiona 'q' para salir.")

    while True:
        ret, frame = camara.read()
        if not ret:
            break

        frame_gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rostros = clasificador_caras.detectMultiScale(
            frame_gris,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        for (x, y, w, h) in rostros:
            rostro_bgr = frame[y:y+h, x:x+w]
            
            if rostro_bgr.size == 0:
                continue

            etiqueta_raw, probabilidad = predecir_clase(rostro_bgr)

            if probabilidad >= UMBRAL_CONFIANZA:
                texto = f"{etiqueta_bonita(etiqueta_raw)} {probabilidad*100:.1f}%"
                color_rectangulo = (0, 255, 0)
            else:
                texto = "Desconocido"
                color_rectangulo = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color_rectangulo, 2)
            cv2.putText(
                frame,
                texto,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color_rectangulo,
                2,
                cv2.LINE_AA
            )

        cv2.imshow("Reconocimiento facial", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camara.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()