import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# [1a] Hilfsfunktion zum Laden der Batches
def load_batch(file):
    """
    Lädt einen einzelnen Batch aus einer Datei
    Args:
        file: Dateipfad zum Batch
    Returns:
        Tuple mit Daten und Labels
    """
    with open(file, 'rb') as f:
        batch = pickle.load(f, encoding='latin1')
    return batch['data'], batch['labels']

# [1a] Laden und Zusammenfügen aller Trainingsdaten
Xtr, Ytr = [], []
for i in range(1, 6):
    data, labels = load_batch('cifar-10-python/cifar-10-batches-py/data_batch_' + str(i))
    Xtr.append(data)
    Ytr += labels
Xtr = np.vstack(Xtr)
Ytr = np.array(Ytr)

# [1b] Laden der Testdaten
Xte, Y = load_batch('cifar-10-python/cifar-10-batches-py/test_batch')
Xte = np.array(Xte)
Y = np.array(Y)

# [1c] Funktion zur Bildvisualisierung
def image(img_vector):
    """
    Konvertiert einen 3072-dimensionalen Vektor in ein 32x32x3 Bild
    Args:
        img_vector: Eindimensionaler Array mit 3072 Elementen
    """
    img = img_vector.reshape(3, 32, 32).transpose(1, 2, 0)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

# [2a] Implementation der Distanzmaße
def l1_distance(x1, x2):
    """L1-Distanzmaß (Manhattan-Distanz)"""
    return np.sum(np.abs(x1 - x2), axis=1)

def l2_distance(x1, x2):
    """L2-Distanzmaß (Euklidische Distanz)"""
    return np.sqrt(np.sum((x1 - x2) ** 2, axis=1))

# [2a,2b,2c] K-Nearest-Neighbor Implementation
def nearestneighbor(Xtr, Ytr, x_query, k=1, distance_metric='l2'):
    """
    K-Nearest-Neighbor Klassifikation
    Args:
        Xtr: Trainingsdaten
        Ytr: Trainingslabels
        x_query: Zu klassifizierendes Beispiel
        k: Anzahl der nächsten Nachbarn
        distance_metric: 'l1' oder 'l2' für Distanzmaß
    """
    if distance_metric == 'l1':
        distances = l1_distance(Xtr, x_query)
    else:
        distances = l2_distance(Xtr, x_query)
    nearestI = np.argsort(distances)[:k]
    label = Ytr[nearestI]
    label_count = Counter(label)
    most_common = label_count.most_common()
    if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
        return label[0]
    else:
        return most_common[0][0]

# [2b,2c] Visualisierung der ersten 10 Testbilder für alle K-Werte und Distanzmaße
beste_configs = [
    ('L1-Distanz', 5),  # Beste L1 Konfiguration (28.00%)
    ('L2-Distanz', 1)  # Beste L2 Konfiguration (29.00%)
]

for i in range(10):
    print(f"\n=== Bild {i + 1} ===")
    image(Xte[i])
    print(f"Wahres Label = {Y[i]}")
    print("\nVorhersagen (Beste Konfigurationen):")

    for dist_name, k in beste_configs:
        dist_metric = 'l1' if dist_name == 'L1-Distanz' else 'l2'
        pred = nearestneighbor(Xtr, Ytr, Xte[i], k=k, distance_metric=dist_metric)
        print(f"{dist_name} (K={k}): Vorhersage = {pred}")

    print("=" * 40)


# [2d,2e] Berechnung und Visualisierung der Genauigkeiten
def genauigkeit(y_echt, y_vorhersage):
    """Berechnet die Klassifikationsgenauigkeit"""
    return np.mean(y_echt == y_vorhersage)

# Test für verschiedene K-Werte und beide Distanzmaße
k_werte = [1, 3, 5, 7]
genauigkeiten_l1 = []
genauigkeiten_l2 = []

for k in k_werte:
    y_pred_l1 = []
    y_pred_l2 = []
    print(f"\n--- K = {k} ---")
    for i in range(100):
        vorhersage_l1 = nearestneighbor(Xtr, Ytr, Xte[i], k=k, distance_metric='l1')
        vorhersage_l2 = nearestneighbor(Xtr, Ytr, Xte[i], k=k, distance_metric='l2')
        y_pred_l1.append(vorhersage_l1)
        y_pred_l2.append(vorhersage_l2)
    
    acc_l1 = genauigkeit(Y[:100], y_pred_l1)
    acc_l2 = genauigkeit(Y[:100], y_pred_l2)
    genauigkeiten_l1.append(acc_l1)
    genauigkeiten_l2.append(acc_l2)
    
    print(f"L1-Distanz Genauigkeit bei K={k}: {acc_l1:.2%}")
    print(f"L2-Distanz Genauigkeit bei K={k}: {acc_l2:.2%}")

# [2d,2e] Visualisierung der Ergebnisse
plt.figure(figsize=(10, 6))
x = np.arange(len(k_werte))
width = 0.35

genauigkeiten_l1_prozent = [g * 100 for g in genauigkeiten_l1]
genauigkeiten_l2_prozent = [g * 100 for g in genauigkeiten_l2]

plt.bar(x - width/2, genauigkeiten_l1_prozent, width, label='L1-Distanz', color='skyblue')
plt.bar(x + width/2, genauigkeiten_l2_prozent, width, label='L2-Distanz', color='lightgreen')

plt.title("Vergleich L1- vs L2-Distanz bei K-Nearest-Neighbor")
plt.xlabel("K-Wert")
plt.ylabel("Genauigkeit [%]")
plt.xticks(x, k_werte)
plt.legend()
plt.grid(axis='y')
plt.ylim(0, 100)

for i in range(len(k_werte)):
    plt.text(i - width/2, genauigkeiten_l1_prozent[i] + 1, f'{genauigkeiten_l1_prozent[i]:.1f}%', 
             ha='center', va='bottom')
    plt.text(i + width/2, genauigkeiten_l2_prozent[i] + 1, f'{genauigkeiten_l2_prozent[i]:.1f}%', 
             ha='center', va='bottom')

plt.show()

# [2e] Auswertung der Ergebnisse
beste_genauigkeit_l1 = max(genauigkeiten_l1)
bester_k_l1 = k_werte[genauigkeiten_l1.index(beste_genauigkeit_l1)]

beste_genauigkeit_l2 = max(genauigkeiten_l2)
bester_k_l2 = k_werte[genauigkeiten_l2.index(beste_genauigkeit_l2)]

print("\nZusammenfassung der Ergebnisse:")
print(f"Beste L1-Genauigkeit: {beste_genauigkeit_l1:.2%} (K={bester_k_l1})")
print(f"Beste L2-Genauigkeit: {beste_genauigkeit_l2:.2%} (K={bester_k_l2})")

if beste_genauigkeit_l1 > beste_genauigkeit_l2:
    print("\nL1-Distanz erzielte bessere Ergebnisse")
elif beste_genauigkeit_l2 > beste_genauigkeit_l1:
    print("\nL2-Distanz erzielte bessere Ergebnisse")
else:
    print("\nBeide Distanzmaße erzielten gleiche Ergebnisse")