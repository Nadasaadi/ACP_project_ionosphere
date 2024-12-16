import tkinter as tk
from tkinter import scrolledtext
from tkinter import messagebox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from utils.acp_steps import normalize_data, compute_correlation_matrix, compute_eigenvalues_eigenvectors

# Charger les données
data = pd.read_csv('data/ionosphere.arff.csv')
numeric_data = data.loc[:, 'a1':'a34'].values

# Normalisation des données
normalized_data = normalize_data(numeric_data)

# Calcul de la matrice de corrélation
correlation_matrix = compute_correlation_matrix(normalized_data)

# Calcul des valeurs propres et vecteurs propres
eigenvalues, eigenvectors = compute_eigenvalues_eigenvectors(correlation_matrix)

# Calcul de la variance expliquée
explained_variance = eigenvalues / np.sum(eigenvalues)
explained_variance_ratio = explained_variance / np.sum(explained_variance) * 100

# Application de t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
tsne_result = tsne.fit_transform(normalized_data)

# Création de la fenêtre principale
root = tk.Tk()
root.title("Analyse en Composantes Principales (ACP)")

# Créer un cadre pour les boutons et les placer à gauche
button_frame = tk.Frame(root)
button_frame.pack(side="left", padx=10, pady=10)

# Titre principal dans l'interface
title_label = tk.Label(root, text="Résultats de l'Analyse en Composantes Principales (ACP)", font=("Helvetica", 16))
title_label.pack(pady=10)

# Zone de texte défilante pour afficher les résultats
result_text = scrolledtext.ScrolledText(root, width=80, height=20, wrap=tk.WORD)
result_text.pack(padx=20, pady=20)

# Label pour afficher le message en vert concernant la variance expliquée
message_variance_label = tk.Label(root, text="", font=("Helvetica", 12))
message_variance_label.pack(pady=10)

# Fonction pour afficher les résultats dans la zone de texte
def afficher_resultats():
    # Effacer le contenu précédent
    result_text.delete(1.0, tk.END)

    message = f"Valeurs propres :\n{eigenvalues}\n\n"
    message += f"Vecteurs propres :\n{eigenvectors}\n\n"
    message += f"Variance expliquée par chaque composante :\n{explained_variance_ratio}\n"
    result_text.insert(tk.END, message)

    # Vérification de la variance cumulée et affichage du message en fonction du seuil
    cumulative_variance = np.cumsum(explained_variance_ratio)
    for i, cum_var in enumerate(cumulative_variance):
        if cum_var > 90:  # Si la variance cumulée dépasse 90%
            message_variance = f"\nLes {i+1} premières composantes principales capturent {cum_var:.2f}% de l'information.\n"
            message_variance_label.config(text=message_variance, fg="green")
            break
    else:
        # Si aucune composante ne dépasse 90%, afficher un message par défaut
        message_variance = f"\nLa variance cumulée n'atteint pas 90%.\n"
        message_variance_label.config(text=message_variance, fg="red")

# Fonction pour afficher la matrice de corrélation
def afficher_matrice_correlation():
    # Effacer le contenu précédent
    result_text.delete(1.0, tk.END)

    # Afficher la matrice de corrélation dans la zone de texte
    message = f"Matrice de corrélation :\n{correlation_matrix}\n"
    result_text.insert(tk.END, message)

# Fonction pour afficher la projection avec t-SNE
def afficher_projection_tsne():
    class_labels = data['class'].astype('category')
    class_codes = class_labels.cat.codes
    class_names = class_labels.cat.categories
    colors = ['red', 'green', 'blue']
    plt.figure(figsize=(8, 6))
    for i, class_name in enumerate(class_names):
        plt.scatter(tsne_result[class_codes == i, 0], 
                    tsne_result[class_codes == i, 1], 
                    label=class_name, 
                    color=colors[i])
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title('Projection des données (t-SNE)')
    plt.legend(title="Classe")
    plt.grid(True)
    plt.show()

# Créer les boutons

# Bouton pour afficher la matrice de corrélation
button_matrice_correlation = tk.Button(button_frame, text="Afficher la matrice de corrélation", command=afficher_matrice_correlation, font=("Helvetica", 12))
button_matrice_correlation.pack(pady=10)

# Bouton pour afficher les résultats ACP
button_afficher = tk.Button(button_frame, text="Afficher les résultats ACP", command=afficher_resultats, font=("Helvetica", 12))
button_afficher.pack(pady=10)

# Bouton pour afficher la projection t-SNE
button_projection_tsne = tk.Button(button_frame, text="Afficher projection t-SNE", command=afficher_projection_tsne, font=("Helvetica", 12))
button_projection_tsne.pack(pady=10)

# Fonction de fermeture de l'application
def fermer_app():
    root.quit()

# Créer un bouton pour fermer l'application
button_quit = tk.Button(button_frame, text="Fermer", command=fermer_app, font=("Helvetica", 12), fg="red")
button_quit.pack(pady=10)

# Lancer l'interface graphique
root.mainloop()
