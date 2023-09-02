from django.db import models

class UploadedFile(models.Model):
    file = models.FileField(upload_to='uploaded_files/')
    
    class Meta:
        app_label = 'mybackend'


class Spectrum(models.Model):
    wavelength = models.FloatField()
    concentration = models.FloatField()
    intensity = models.FloatField()
