from django.shortcuts import render
import requests
from rest_framework import generics
from rest_framework.response import Response
from .serializers import RequestSerializer, PredictionSerializer

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.base")
import os
import numpy as np
import joblib
import spacy

# Import the required libraries
from google.cloud import translate_v2 as translate


# Create your views here.

def extract_symptoms(text):
    # Load the NER model
    nlp = spacy.load("models/model-best")
    doc = nlp(text)
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

            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'mainapp/balmy-link-383016-9fb29fcd60f2.json'
            # Set up the translation client
            translate_client = translate.Client()
            # Set the source and target languages
            source_lang = 'si'
            target_lang = 'en'
            # Define the input text
            input_text = text
            # Use the translation API to translate the input text
            result = translate_client.translate(input_text, source_language='si', target_language='en')
            # result = translate_client.translate(
            #     text, source_language='en', target_language=target)
            # Print the translated text
            print('Input text: ', input_text)
            print('Translated text: ', result['translatedText'])
            extracted_symptoms = extract_symptoms(result['translatedText'])
            print(extracted_symptoms)

            data = {
                'symptoms': extracted_symptoms,
            }

            return Response(data=data)
        else:
            extracted_symptoms = extract_symptoms(text)
            print(extracted_symptoms)

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

        """## **Disease Prediction**"""

        # List of all possible symptoms
        symptoms_keys = ['COUGH', 'MUSCLE_ACHES', 'TIREDNESS', 'SORE_THROAT', 'RUNNY_NOSE', 'STUFFY_NOSE',
                         'FEVER', 'NAUSEA', 'VOMITING', 'DIARRHEA', 'SHORTNESS_OF_BREATH', 'DIFFICULTY_BREATHING',
                         'LOSS_OF_TASTE', 'LOSS_OF_SMELL', 'ITCHY_NOSE', 'ITCHY_EYES', 'ITCHY_MOUTH', 'ITCHY_INNER_EAR',
                         'SNEEZING', 'PINK_EYE',
                         'SKIN_RASH', 'CHILLS', 'jOINT_PAIN', 'FATIGUE', 'HEAD_ACHE', 'LOSS_OF_APPETITES',
                         'PAIN_BEHIND_THE_EYES', 'BACK_PAIN',
                         'MALAISE', 'RED_SPOTS_OVER_BODY']
        # Input symptoms as text
        input_symptoms = symptoms

        # Create an empty array to store the binary values
        binary_input = np.zeros(len(symptoms_keys))

        # Iterate over the input symptoms and set the corresponding element in the binary array to 1
        for symptom in input_symptoms:
            if symptom in symptoms_keys:
                binary_input[symptoms_keys.index(symptom)] = 1

        # Load the Logistic Regression model
        clf = joblib.load("models/RandomForestClassifier.pkl")

        # Make predictions on the binary input data
        predictions = clf.predict(binary_input.reshape(1, -1))
        print(binary_input.reshape(1, -1))
        # The output will be the predicted disease
        print(predictions)

        data = {
            'symptoms': symptoms,
            'prediction': predictions
        }

        return Response(data=data)
