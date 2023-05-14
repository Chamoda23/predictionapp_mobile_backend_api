from google.cloud import translate_v2 as translate
import spacy
import joblib
import numpy as np
import os
from django.shortcuts import render
from rest_framework import generics
from rest_framework.response import Response
from .serializers import RequestSerializer, PredictionSerializer
import warnings
from spellchecker import SpellChecker

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.base")

# Import the required libraries


# Create your views here.


def extract_symptoms(text):
    # Load the NER model
    spell = SpellChecker()
    spell.word_frequency.load_words(['COUGH','MUSCLE_ACHES','TIREDNESS','SORE_THROAT','RUNNY_NOSE','STUFFY_NOSE','FEVER','NAUSEA','VOMITING','DIARRHEA','SHORTNESS_OF_BREATH','DIFFICULTY_BREATHING','LOSS_OF_TASTE','LOSS_OF_SMELL','ITCHY_NOSE','ITCHY_EYES','ITCHY_MOUTH','ITCHY_INNER_EAR','SNEEZING','PINK_EYE','SKIN_RASH', 'CHILLS', 'jOINT_PAIN', 'FATIGUE', 'HEADACHE', 'LOSS_OF_APPETITES', 'PAIN_BEHIND_THE_EYES', 'BACK_PAIN','MALAISE', 'RED_SPOTS_OVER_BODY'])
    phrases_to_remove = ["I have a", "I have"]
    for phrase in phrases_to_remove:
        text = text.replace(phrase, "")
    symptoms = [symptom.strip().replace(' ', '_') for symptom in text.split(',')]
    corrected_symptoms = []
    for symptom in symptoms:
        correction = spell.correction(symptom)
        if correction is not None:
            corrected_symptoms.append(correction.replace('_', ' '))

    corrected_text = ', '.join(corrected_symptoms)
    nlp = spacy.load("models/model-best")
    doc = nlp(corrected_text)
    extracted_symptoms = []
    for ent in doc.ents:
        if ent.label_ == "SYMPTOMS":  # and ent.text.lower() in symptoms:
            extracted_symptoms.append(ent.text.upper().replace(" ", "_"))
    return extracted_symptoms


class RequestView(generics.CreateAPIView):
    serializer_class = RequestSerializer

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        text = serializer.validated_data['text']
        is_sinhala = serializer.validated_data['is_sinhala']

        if is_sinhala is True:

            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'mainapp/balmy-link-383016-eef88c637cc9.json'
            # Set up the translation client
            translate_client = translate.Client()
            # Set the source and target languages
            source_lang = 'si'
            target_lang = 'en'
            # Define the input text
            input_text = text
            # Use the translation API to translate the input text
            result = translate_client.translate(
                input_text, source_language='si', target_language='en')
            print('Input text: ', input_text)
            print('Translated text: ', result['translatedText'])
            extracted_symptoms = extract_symptoms(result['translatedText'])

            data = {
                'symptoms': extracted_symptoms,
            }

            return Response(data=data)
        else:
            extracted_symptoms = extract_symptoms(text)

            data = {
                'symptoms': extracted_symptoms,
            }

            return Response(data=data)


class PredictView(generics.CreateAPIView):
    serializer_class = PredictionSerializer

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        text = serializer.validated_data['text']
        symptoms = serializer.validated_data['symptoms']
        if len(symptoms) == 0:
            extracted_symptoms = self._extract_symptoms(text)
        else:
            extracted_symptoms = symptoms

        """## **Disease Prediction**"""

        input_symptoms = eval(extracted_symptoms)
        # List of all possible symptoms
        symptoms_keys = ['COUGH', 'MUSCLE_ACHES', 'TIREDNESS', 'SORE_THROAT', 'RUNNY_NOSE', 'STUFFY_NOSE',
                         'FEVER', 'NAUSEA', 'VOMITING', 'DIARRHEA', 'SHORTNESS_OF_BREATH', 'DIFFICULTY_BREATHING',
                         'LOSS_OF_TASTE', 'LOSS_OF_SMELL', 'ITCHY_NOSE', 'ITCHY_EYES', 'ITCHY_MOUTH', 'ITCHY_INNER_EAR',
                         'SNEEZING', 'PINK_EYE', 'SKIN_RASH', 'CHILLS', 'jOINT_PAIN', 'FATIGUE', 'HEADACHE', 'LOSS_OF_APPETITES',
                         'PAIN_BEHIND_THE_EYES', 'BACK_PAIN', 'MALAISE', 'RED_SPOTS_OVER_BODY']

        # Create an empty array to store the binary values
        binary_input = np.zeros(len(symptoms_keys))
        for symptom in input_symptoms:
            if symptom in symptoms_keys:
                binary_input[symptoms_keys.index(symptom)] = 1

        best_model = joblib.load("models/best_model.jolib")
        probabilities = best_model.predict_proba(binary_input.reshape(1, -1))
        disease_probabilities = {}
        for i, disease in enumerate(best_model.classes_):
            disease_probabilities[disease] = round(
                probabilities[0][i] * 100, 2)

        highest_prob_disease = max(
            disease_probabilities, key=disease_probabilities.get)
        highest_prob_percentage = disease_probabilities[highest_prob_disease]
        print(highest_prob_percentage)
        print(highest_prob_disease)

        data = {
            'symptoms': symptoms,
            'prediction': "{} - {}%".format(highest_prob_disease, highest_prob_percentage)
        }
        return Response(data=data)
