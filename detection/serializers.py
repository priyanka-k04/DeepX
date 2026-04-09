from rest_framework import serializers
from .models import AnalysisResult

class AnalysisResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = AnalysisResult
        # This will include all fields: file_name, media_type, prediction, confidence, timestamp
        fields = '__all__'