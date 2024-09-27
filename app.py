import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# Streamlit app title
st.title("Distillation Binaire : Méthode de McCabe - Thiele")

# Input parameters
xF = st.number_input("Fraction molaire du composant léger dans l'alimentation (xF)", value=0.10, min_value=0.0, max_value=1.0)
xD = st.number_input("Fraction molaire du composant léger dans le distillat (xD)", value=0.80, min_value=0.0, max_value=1.0)
xW = st.number_input("Fraction molaire du composant léger dans le rebouilleur (xW)", value=0.10, min_value=0.0, max_value=1.0)
R_factor = st.number_input("Facteur du ratio de reflux (L/D)", value=1.5, min_value=0.0)
a = st.number_input("Facteur de séparation (a)", value=2.5, min_value=0.0)
q = st.number_input("Condition d'alimentation (q)", value=1.5, min_value=0.0)

# Check for valid q value
if q <= 0 or q == 1:
    st.error("La condition d'alimentation (q) doit être supérieure à 0 et différente de 1.")
else:
    # Courbe d'équilibre
    def eq_curve(a):
        x_eq = np.linspace(0, 1, 51)
        y_eq = a * x_eq / (1 + (a - 1) * x_eq)
        return y_eq, x_eq

    y_eq, x_eq = eq_curve(a)

    # Ligne d'alimentation
    def fed(xF, q, a):    
        c1 = (q * (a - 1))
        c2 = q + xF * (1 - a) - a * (q - 1)
        c3 = -xF
        coeff = [c1, c2, c3]
        r = np.sort(np.roots(coeff))
        
        xiE = r[0] if r[0] > 0 else r[1]
        yiE = a * xiE / (1 + xiE * (a - 1))
        
        if q == 1:
            x_fed = [xF, xF]
            y_fed = [xF, yiE]
        else:
            x_fed = np.linspace(xF, xiE, 51)
            y_fed = q / (q - 1) * x_fed - xF / (q - 1)
        
        return xiE, yiE, y_fed, x_fed

    xiE, yiE, y_fed, x_fed = fed(xF, q, a)

    # Calcul de R_min et R
    R_min = (xD - yiE) / (yiE - xiE)
    R = R_factor * R_min

    # Point d'alimentation
    xiF = (xF / (q - 1) + xD / (R + 1)) / (q / (q - 1) - R / (R + 1))
    yiF = R / (R + 1) * xiF + xD / (R + 1)

    # Section de rectification
    def rect(R, xD, xiF):
        x_rect = np.linspace(xiF - 0.025, xD, 51)    
        y_rect = R / (R + 1) * x_rect + xD / (R + 1)
        return y_rect, x_rect

    y_rect, x_rect = rect(R, xD, xiF)

    # Section de stripping
    def stp(xiF, yiF, xW):
        x_stp = np.linspace(xW, xiF + 0.025, 51)    
        y_stp = ((yiF - xW) / (xiF - xW)) * (x_stp - xW) + xW
        return y_stp, x_stp

    y_stp, x_stp = stp(xiF, yiF, xW)

    # Construction des étages
    s = np.zeros((1000, 5))
    for i in range(1, 1000):
        s[0, 0] = xD
        s[0, 1] = xD
        s[0, 2] = s[0, 1] / (a - s[0, 1] * (a - 1))
        s[0, 3] = s[0, 1]
        s[0, 4] = 0
        
        s[i, 0] = s[i - 1, 2]
        
        if s[i, 0] < xW:
            s[i, 1] = s[i, 0] 
            s[i, 2] = s[i, 0]
            s[i, 3] = s[i, 0]
            s[i, 4] = i
            break  
        if s[i, 0] > xiF:
            s[i, 1] = R / (R + 1) * s[i, 0] + xD / (R + 1)
        elif s[i, 0] < xiF:
            s[i, 1] = ((yiF - xW) / (xiF - xW)) * (s[i, 0] - xW) + xW
        else:
            s[i, 1] = s[i - 1, 3]
        
        if s[i, 0] > xW:
            s[i, 2] = s[i, 1] / (a - s[i, 1] * (a - 1))
        else:
            s[i, 2] = s[i, 0]
        
        s[i, 3] = s[i, 1]
        
        if s[i, 0] < xiF:
            s[i, 4] = i
        else:
            s[i, 4] = 0

    s = s[~np.all(s == 0, axis=1)]
    s_rows = np.size(s, 0)
    S = np.zeros((s_rows * 2, 2))

    for i in range(0, s_rows):
        S[i * 2, 0] = s[i, 0]
        S[i * 2, 1] = s[i, 1]
        S[i * 2 + 1, 0] = s[i, 2]
        S[i * 2 + 1, 1] = s[i, 3]

    # Numérotation des étages
    x_s = s[:, 2:3]
    y_s = s[:, 3:4]
    stage = np.char.mod('%d', np.linspace(1, s_rows - 1, s_rows - 1))

    # Emplacement de la plaque d'alimentation
    s_f = s_rows - np.count_nonzero(s[:, 4:5], axis=0)
    s_f_scalar = s_f.item()

    # Tracé
    fig, ax = plt.subplots(figsize=(7, 6), dpi=600)
    ax.plot([0, 1], [0, 1], "k-")
    ax.plot(x_eq, y_eq, "r-", label="Courbe d'équilibre")
    ax.plot(x_rect, y_rect, 'k--', label="Section de rectification OL")
    ax.plot(x_stp, y_stp, 'k-.', label="Section de stripping OL")
    ax.plot(x_fed, y_fed, 'k:', label="Ligne d'alimentation")
    ax.plot(S[:, 0], S[:, 1], 'b-', label="Étages")

    # Numéros d'étages
    for label, x, y in zip(stage, x_s, y_s):
        ax.annotate(label, xy=(x, y), xytext=(0, 5), textcoords='offset points', ha='right')

    # Points d'alimentation, de distillat et de rebouilleur
    ax.plot(xF, xF, 'go', markersize=5)
    ax.plot(xD, xD, 'go', markersize=5)
    ax.plot(xW, xW, 'go', markersize=5)
    ax.text(xF + 0.05, xF - 0.03, '($x_{F}, x_{F}$)', horizontalalignment='center')
    ax.text(xD + 0.05, xD - 0.03, '($x_{D}, x_{D}$)', horizontalalignment='center')
    ax.text(xW + 0.05, xW - 0.03, '($x_{W}, x_{W}$)', horizontalalignment='center')

    # Intersection: Rectifying + Stripping + Feedline
    ax.plot(xiF, yiF, 'go', markersize=5)
    ax.text(xiF + 0.05, yiF - 0.03, '($x_{F}, y_{F}$)', horizontalalignment='center')

    # Configuration du tracé
    ax.set_xlabel("Fraction molaire du composant léger (x)")
    ax.set_ylabel("Fraction molaire du composant léger (y)")
    ax.set_title("Diagramme de McCabe-Thiele")
    ax.legend()
    ax.grid()

    # Affichage des résultats
    st.pyplot(fig)
    st.write(f"**Rmin**: {R_min:.2f}")
    st.write(f"**R**: {R:.2f}")
    st.write(f"**Nombre d'étages**: {s_rows - 1}")
    st.write(f"**Étages de l'alimentation**: {s_f_scalar}")
