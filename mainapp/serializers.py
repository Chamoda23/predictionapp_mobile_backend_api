from rest_framework import serializers


class RequestSerializer(serializers.Serializer):
    text = serializers.CharField(max_length=1000)
    is_sinhala = serializers.BooleanField(default=False)


class PredictionSerializer(serializers.Serializer):
    text = serializers.CharField(max_length=1000)
    symptoms = serializers.CharField(max_length=1000)
