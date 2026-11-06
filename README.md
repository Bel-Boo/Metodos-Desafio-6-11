# 🐾 Refugio Huellitas – EDO con Heun y RK4

Desafío individual de Métodos Numéricos · Autor: [TU NOMBRE COMPLETO]

## 🌐 Probar la aplicación (sin instalar nada)
👉 **[PEGA AQUÍ EL LINK DE STREAMLIT]**

Se abre directo en el navegador.

## Descripción del modelo
Población de un refugio de mascotas:

    dy/dt = a - k·y ,  y(0) = y0

Solución analítica: y(t) = a/k + (y0 - a/k)·e^(-kt)

- y(t): animales en el refugio (animales) · t: tiempo (semanas)
- a: ingreso de animales (animales/semana) · k: tasa de adopción (1/semana)
- Los parámetros son **hipotéticos**; no es una predicción con datos reales.

## Qué incluye la app
Heun y RK4 con el mismo h, solución analítica verificada, tabla por pasos, gráficas,
errores absoluto y relativo, comparación de dos tamaños de paso, orden de convergencia
e interpretación automática.

## Ejecución local (opcional)
Solo si quieres correr el código en tu computador:

    pip install -r requirements.txt
    streamlit run app.py
