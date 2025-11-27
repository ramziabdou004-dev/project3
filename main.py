import cv2
import numpy as np
import streamlit as st
from keras.applications.mobilenet_v2 import (MobileNetV2, preprocess_input, decode_predictions)
from PIL import Image

# 1. Correction: Ajout du décorateur @st.cache_resource ici pour charger le modèle une seule fois.
@st.cache_resource 
def load_model():
    """Charge le modèle MobileNetV2 pré-entraîné sur ImageNet."""
    # Correction 1: 'weights' est au pluriel dans la plupart des versions modernes de Keras.
    model = MobileNetV2(weights='imagenet')
    return model

def image_propres(image: Image.Image):
    """Prétraite l'objet PIL Image pour qu'il corresponde au format MobileNetV2 (224x224, normalisé)."""
    # 2. Correction: PIL Image n'a pas de fonction .array(). On doit utiliser np.array(img).
    # De plus, cv2.resize attend l'image comme un tableau NumPy.
    img = np.array(image)
    
    # img est (H, W, C) après np.array(image)
    # Correction 3: cv2.resize prend une tupple pour la taille, soit (224, 224)
    img = cv2.resize(img, (224, 224))
    
    # img est maintenant (224, 224, 3)
    img = preprocess_input(img)
    # img est normalisé
    img = np.expand_dims(img, axis=0)
    # img est (1, 224, 224, 3)
    return img

def classifiy(model, image: Image.Image):
    """Prédit la classification de l'image."""
    try:
        process = image_propres(image)
        # S'assurer que model est bien l'objet Keras, pas la fonction
        prediction = model.predict(process)
        # La fonction retourne le résultat décodé
        decode_result = decode_predictions(prediction, top=3)[0]
        
        # Correction 4: Retourner la variable qui contient le résultat, pas la fonction
        return decode_result
        
    except Exception as e:
        # st.error est appelé dans le bloc exception, ce qui est correct.
        st.error(f'Erreur lors de la classification de l\'image: {str(e)}')
        return None
    
def main():
    st.set_page_config(page_title='Classification Images avec IA', page_icon='🕵️‍♀️', layout='centered')
    st.title('AI Image Classifier (MobileNetV2)')
    st.write("Chargez une image et laissez l'IA vous dire ce qu'elle voit.")
    
    # Le chargement du modèle doit être fait correctement
    model = load_model() # Cette ligne appelle la fonction mise en cache
    
    upload = st.file_uploader('Choisissez une image', type=['jpg', 'png', 'jpeg'])
    
    if upload is not None:
        # Ouvrir l'image une seule fois avec PIL
        image_pil = Image.open(upload)
        
        # Afficher l'image (le bouton 'st.image' retourne None, on utilise l'objet image_pil)
        st.image(image_pil, caption='Image Chargée', use_container_width=True)

        btn = st.button('Classer l\'Image')
        
        if btn:
            with st.spinner('Analyse de l\'image en cours...'):
                # Correction 5: Utiliser l'objet PIL Image.open(upload) déjà chargé
                prediction = classifiy(model, image_pil)
                
                if prediction:
                    st.subheader('Résultats de la Prédiction')
                    
                    # On boucle sur la structure décodée: (ID, label, score)
                    for _, label, score in prediction:
                        st.write(f"**{label.capitalize()}**: {score:.2%}")

if __name__ == '__main__':
    main()