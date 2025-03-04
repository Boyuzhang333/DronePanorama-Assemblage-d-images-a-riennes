

## 2025/03/04
## 1. Introduction
Dans le cadre de la surveillance des herbiers de Posidonie, plusieurs technologies ont été étudiées pour garantir une extraction et une correspondance précises des caractéristiques des images prises par drone.

## 2. Extraction des Caractéristiques

### SIFT (Scale-Invariant Feature Transform)
La méthode **SIFT** est utilisée pour détecter des points caractéristiques invariants aux variations d'échelle, de rotation, de lumière et d'affinité.  
Elle utilise la **pyramide gaussienne** et le **DOG (Difference of Gaussian)** pour identifier des points stables dans différentes échelles d'images.

#### Propriétés de SIFT :
- **Invariance à l'échelle** : Permet de détecter les points caractéristiques indépendamment de la distance.
- **Invariance à la rotation** : Les points caractéristiques restent identifiables même si l’image est tournée.
- **Invariance à la lumière** : Les variations de lumière n’affectent pas l’identification des caractéristiques.
- **Invariance aux transformations affines** : Les déformations de l’image ne modifient pas la capacité de détection.

## 3. Correspondance des Points

### BFMatcher (Brute Force Matcher)
Après l'extraction des caractéristiques, **BFMatcher** est utilisé pour comparer les descripteurs des points clés entre les images.  
Cette technique permet d’évaluer la similarité des points à travers des méthodes de calcul de distance, en prenant en compte les descripteurs binaires et flottants.

## 4. Stratégie d'Acquisition par Drone

### Paramètres Environnementaux
Les facteurs suivants sont cruciaux pour une acquisition d'images réussie :
- **Lumière**
- **Vents marins**
- **Conditions météorologiques**

### Paramètres Techniques
Les paramètres suivants influencent la qualité des images collectées :
- **Hauteur de vol**
- **Fréquence d'échantillonnage**
- **Trajectoire de vol**

### Métadonnées
Les métadonnées utilisées pour optimiser la collecte et réduire les erreurs incluent :
- **Horodatage**
- **Ordre de capture**
- **Coordonnées GPS**

## 5. Méthodes de Détection

### Méthodes Comparées
- **SURF**, **ORB** et **BRISK** ont été considérées, mais elles sont moins précises que **SIFT**.  
- Bien qu'elles soient plus rapides, elles ne conviennent pas pour des applications nécessitant une grande précision, comme la création d'images de grande taille pour une analyse détaillée.

## 6. Fusion d'Images

### Matrice d'Homographie
La **matrice d'homographie** permet de transformer les coordonnées d’une image sur une autre pour les aligner et fusionner plusieurs images en une seule image panoramique, minimisant ainsi la distorsion.

## 7. Avancement

La lecture complète du premier rapport a été effectuée, permettant une compréhension détaillée des technologies liées à l'extraction des caractéristiques, la correspondance des points et la fusion d'images.

## 8. Conclusion

Les technologies explorées, telles que **SIFT** et **BFMatcher**, assurent une surveillance précise des herbiers de Posidonie.  
L'approche se concentre sur la qualité des images, en optimisant les paramètres d’acquisition et en intégrant les métadonnées pour un traitement fiable des données.

