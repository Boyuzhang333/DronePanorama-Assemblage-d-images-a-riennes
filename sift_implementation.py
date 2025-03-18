import cv2
import numpy as np
import time

def extract_sift_features(image):
    """
    Extrait les caractéristiques SIFT d'une image
    
    Args:
        image: Image d'entrée (format numpy array)
    
    Returns:
        keypoints: Points caractéristiques détectés
        descriptors: Descripteurs des points caractéristiques
    """
    # Conversion en niveaux de gris si nécessaire
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Création du détecteur SIFT
    sift = cv2.SIFT_create(
        nfeatures=0,  # Nombre maximum de caractéristiques (0 = illimité)
        nOctaveLayers=3,  # Nombre de couches par octave
        contrastThreshold=0.04,  # Seuil de contraste
        edgeThreshold=10,  # Seuil de bord
        sigma=1.6  # Sigma pour le flou gaussien
    )
    
    # Détection des points caractéristiques et calcul des descripteurs
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    return keypoints, descriptors

def visualize_keypoints(image, keypoints):
    """
    Visualise les points caractéristiques sur l'image
    
    Args:
        image: Image d'entrée
        keypoints: Points caractéristiques à visualiser
    
    Returns:
        Image avec les points caractéristiques visualisés
    """
    # Copie de l'image pour ne pas la modifier
    img = image.copy()
    
    # Dessin des points caractéristiques
    cv2.drawKeypoints(
        image,
        keypoints,
        img,
        color=(0, 255, 0),  # Couleur verte
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    
    return img

def test_sift_extraction(image_path):
    """
    Fonction de test pour l'extraction SIFT
    
    Args:
        image_path: Chemin vers l'image de test
    """
    # Chargement de l'image de test
    image = cv2.imread(image_path)
    if image is None:
        print(f"Erreur: Impossible de charger l'image {image_path}")
        return
    
    # Mesure du temps de traitement
    start_time = time.time()
    
    # Extraction des caractéristiques
    keypoints, descriptors = extract_sift_features(image)
    
    # Calcul du temps de traitement
    processing_time = time.time() - start_time
    
    # Affichage des statistiques
    print(f"Nombre de points caractéristiques détectés: {len(keypoints)}")
    print(f"Temps de traitement: {processing_time:.2f} secondes")
    
    # Visualisation des résultats
    result = visualize_keypoints(image, keypoints)
    
    # Affichage des résultats
    cv2.imshow('Points caractéristiques SIFT', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Test avec une image
    test_sift_extraction("test_image.jpg") 