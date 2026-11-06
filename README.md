# 🐾 Refugio Huellitas – EDO con Heun y RK4

Desafío individual de Métodos Numéricos. Modela la población de un refugio de mascotas con

    dy/dt = a - k·y ,  y(0) = y0

Solución analítica: y(t) = a/k + (y0 - a/k)·e^(-kt)

- y(t): animales en el refugio (animales) · t: semanas
- a: ingreso (animales/semana) · k: tasa de adopción (1/semana)
- Los parámetros son **hipotéticos**; no es una predicción con datos reales.

## Ejecutar
    pip install -r requirements.txt
    streamlit run app.py

## Qué incluye
Heun y RK4 con el mismo h, solución analítica verificada, tabla por pasos, gráficas,
errores absoluto y relativo, comparación de dos tamaños de paso, orden de convergencia
e interpretación automática.
