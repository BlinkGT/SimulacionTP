import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io
import tempfile
import os
from PIL import Image # Necesario para crear GIF directamente

# --- Constantes físicas ---
G = 9.81  # Aceleración debido a la gravedad (m/s^2)

# --- Configuración de Streamlit ---
st.set_page_config(
    page_title="Simulador de Tiro Parabólico",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Simulador Interactivo de Tiro Parabólico")
st.markdown("""
    Explora la trayectoria de un proyectil variando su velocidad inicial y ángulo de lanzamiento.
    Observa la altura máxima, el alcance y la animación del disparo.
""")

# --- Controles de usuario (barra lateral) ---
with st.sidebar:
    st.header("Parámetros del Disparo")
    
    velocidad_inicial = st.slider(
        "Velocidad Inicial (m/s)",
        min_value=10,
        max_value=100,
        value=50,
        step=5,
        help="La velocidad con la que el proyectil es lanzado."
    )
    
    angulo_grados = st.slider(
        "Ángulo de Lanzamiento (grados)",
        min_value=0,
        max_value=90,
        value=45,
        step=1,
        help="El ángulo sobre la horizontal con el que el proyectil es lanzado."
    )

    mostrar_animacion = st.checkbox("Mostrar Animación", value=True)
    
    if mostrar_animacion:
        fps_animacion = st.slider(
            "Velocidad de Animación (FPS)",
            min_value=5,
            max_value=60,
            value=30,
            step=5,
            help="Frames por segundo de la animación. Más alto = más rápido y suave."
        )
        
        calidad_animacion = st.slider(
            "Calidad de Animación (DPI)",
            min_value=50,
            max_value=200,
            value=100,
            step=25,
            help="Puntos por pulgada. Más alto = mejor calidad de imagen, pero archivo más grande."
        )
    else:
        fps_animacion = 30 
        calidad_animacion = 100

    st.markdown("---")
    st.info("Ajusta los parámetros y la simulación se actualizará automáticamente.")

# --- Cálculos del Tiro Parabólico ---
angulo_radianes = np.deg2rad(angulo_grados)

if np.sin(angulo_radianes) > 0:
    tiempo_vuelo = (2 * velocidad_inicial * np.sin(angulo_radianes)) / G
else:
    tiempo_vuelo = 0

alcance = (velocidad_inicial**2 * np.sin(2 * angulo_radianes)) / G
altura_maxima = (velocidad_inicial**2 * np.sin(angulo_radianes)**2) / (2 * G)

st.subheader("Resultados del Disparo")
col1, col2 = st.columns(2)
with col1:
    st.metric("Altura Máxima", f"{altura_maxima:.2f} m")
with col2:
    st.metric("Alcance Horizontal", f"{alcance:.2f} m")
st.metric("Tiempo Total de Vuelo", f"{tiempo_vuelo:.2f} s")


# --- Generación de la Trayectoria (para gráfico y animación) ---
# Número de puntos para la trayectoria, asegurando un mínimo para la animación.
BASE_NUM_POINTS_TRAJECTORY = 500 

if tiempo_vuelo > 0:
    tiempo_puntos = np.linspace(0, tiempo_vuelo, BASE_NUM_POINTS_TRAJECTORY)
else:
    tiempo_puntos = np.array([0]) # Si tiempo de vuelo es 0, solo el origen

x_pos = velocidad_inicial * np.cos(angulo_radianes) * tiempo_puntos
y_pos = (velocidad_inicial * np.sin(angulo_radianes) * tiempo_puntos) - (0.5 * G * tiempo_puntos**2)

y_pos[y_pos < 0] = 0 # Asegurarse que la altura no sea negativa al final

# --- Visualización (Gráfico Estático) ---
st.subheader("Gráfico de la Trayectoria")

fig_static, ax_static = plt.subplots(figsize=(10, 6))
ax_static.plot(x_pos, y_pos, color='red', linestyle='--', linewidth=2, label='Trayectoria')

ax_static.plot(0, 0, 'go', markersize=8, label='Punto de Lanzamiento')
if alcance > 0 and tiempo_vuelo > 0:
    ax_static.plot(alcance, 0, 'rx', markersize=10, label='Punto de Impacto')
if altura_maxima > 0:
    idx_max_altura = np.argmax(y_pos)
    ax_static.plot(x_pos[idx_max_altura], y_pos[idx_max_altura], 'b^', markersize=8, label='Altura Máxima')

ax_static.set_xlabel("Distancia Horizontal (m)")
ax_static.set_ylabel("Altura (m)")
ax_static.set_title("Trayectoria del Proyectil")
ax_static.grid(True, linestyle='--', alpha=0.7)
ax_static.set_aspect('equal', adjustable='box')
ax_static.set_xlim(0, max(alcance * 1.1, 10))
ax_static.set_ylim(0, max(altura_maxima * 1.2, 10))
ax_static.legend()
ax_static.set_facecolor("#e0f2f7")
fig_static.patch.set_facecolor("#f0f8ff")

st.pyplot(fig_static)
plt.close(fig_static)

# --- Animación ---
if mostrar_animacion:
    st.subheader("Animación del Disparo")
    
    if len(x_pos) < 2:
        st.warning("No hay suficientes puntos en la trayectoria para generar una animación. "
                   "Intenta con un ángulo y/o velocidad inicial diferentes (ej. Velocidad inicial 50, Ángulo 45) para ver el movimiento.")
    else:
        num_animation_frames = int(min(len(x_pos), tiempo_vuelo * fps_animacion))
        if num_animation_frames < 2: 
             num_animation_frames = 2

        animation_indices = np.linspace(0, len(x_pos) - 1, num_animation_frames, dtype=int)
        
        frames_list = []

        with st.spinner("Generando animación... Esto puede tardar unos segundos."):
            try:
                for i in animation_indices:
                    fig_anim_frame, ax_anim_frame = plt.subplots(figsize=(10, 6), dpi=calidad_animacion)
                    
                    # Configurar límites del gráfico para la animación
                    ax_anim_frame.set_xlim(0, max(alcance * 1.1, 10))
                    ax_anim_frame.set_ylim(0, max(altura_maxima * 1.2, 10))
                    ax_anim_frame.set_xlabel("Distancia Horizontal (m)")
                    ax_anim_frame.set_ylabel("Altura (m)")
                    ax_anim_frame.set_title("Animación de Tiro Parabólico")
                    ax_anim_frame.grid(True, linestyle='--', alpha=0.7)
                    ax_anim_frame.set_aspect('equal', adjustable='box')
                    ax_anim_frame.set_facecolor("#e0f2f7")
                    fig_anim_frame.patch.set_facecolor("#f0f8ff")

                    ax_anim_frame.plot(x_pos[:i+1], y_pos[:i+1], 'o-', color='blue', lw=2, label='Trayectoria actual')
                    ax_anim_frame.plot(x_pos[i], y_pos[i], 'o', color='red', markersize=10, label='Proyectil')

                    buf = io.BytesIO()
                    fig_anim_frame.savefig(buf, format='png', dpi=calidad_animacion)
                    buf.seek(0)
                    
                    # --- CORRECCIÓN CLAVE PARA "QUANTIZATION ERROR" ---
                    # Abrir la imagen, convertirla a modo 'P' (paleta) para GIF
                    img = Image.open(buf)
                    img = img.convert('P', palette=Image.ADAPTIVE) # Adaptar la paleta de colores
                    frames_list.append(img)
                    # --------------------------------------------------

                    plt.close(fig_anim_frame)

                if frames_list:
                    output_gif_buffer = io.BytesIO()
                    frames_list[0].save(
                        output_gif_buffer,
                        format="GIF",
                        save_all=True,
                        append_images=frames_list[1:],
                        duration=int(1000 / fps_animacion),
                        loop=0 
                    )
                    output_gif_buffer.seek(0)
                    st.image(output_gif_buffer.read(), use_container_width=True, caption="Simulación Animada")
                    st.success("Animación generada. ¡Ajusta los parámetros para ver más!")
                else:
                    st.warning("No se pudieron generar frames para la animación. Ajusta los parámetros.")

            except Exception as e:
                st.error(f"Error al generar la animación: {e}")
                st.warning("Asegúrate de que tus parámetros permitan una trayectoria visible. "
                           "Este error puede ocurrir por valores extremos o problemas con la memoria.")
            finally:
                pass 
else:
    st.info("La animación está desactivada. Actívala para ver la trayectoria en movimiento.")

st.markdown("---")
st.write("Creado con Streamlit, NumPy y Matplotlib para fines educativos.")