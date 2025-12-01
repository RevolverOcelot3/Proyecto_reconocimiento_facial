Proyecto de reconocimiento facial en tiempo real usando:

- Python
- TensorFlow / Keras (MobileNetV2 con transfer learning)
- OpenCV (detección de rostros con Haar Cascade)
- Modelo entrenado en Google Colab con un dataset personal de amigos y familia.

-Archivos principales

- `reconocimiento.py` → Script principal, abre la webcam y reconoce las caras.
- `reconocimientoentrenado.keras` → Modelo entrenado.
- `clases.json` → Lista de nombres de las clases (carpetas del dataset).

-Cómo usar

```bash
python reconocimiento.py
