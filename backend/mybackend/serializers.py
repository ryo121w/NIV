from rest_framework import serializers
from .models import UploadedFile
from .models import Spectrum

class UploadedFileSerializer(serializers.ModelSerializer):
    file = serializers.FileField()

    class Meta:
        model = UploadedFile
        fields = "__all__"


class SpectrumSerializer(serializers.ModelSerializer):
    class Meta:
        model = Spectrum
        fields = ['wavelength', 'concentration', 'intensity']