from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views  # この行が重要です
from .views import FileUploadView, SpectrumViewSet
from .views import DifferenceGraphView
from .views import SecondDerivativeGraphView
from .views import dynamic_graph_view
from .views import SaveMolarAbsorptivityView
from .views import GetSavedFilePathView 




router = DefaultRouter()
router.register(r'spectrums', SpectrumViewSet)


urlpatterns = [
    path('api/', include(router.urls)),
    path('api/upload/', FileUploadView.as_view(), name='file-upload'),
    path('api/concentration_graph/', views.ConcentrationGraphView.as_view(), name='concentration-graph'),
    path('api/difference_graph/', DifferenceGraphView.as_view(), name='difference-graph'),
    path('api/second_derivative_graph/', SecondDerivativeGraphView.as_view(), name='second_derivative_graph'),
    path('api/dynamic_graph', dynamic_graph_view, name='dynamic_graph'),
    path('api/find_peaks/', views.find_peaks, name='find_peaks'),
    path('api/calculate_hb_strength/', views.calculate_hb_strength, name='calculate_hb_strength'),
    path('api/save_molar_absorptivity/', SaveMolarAbsorptivityView.as_view(), name='save_molar_absorptivity'),
    path('api/get_saved_file_path/', views.GetSavedFilePathView.as_view(), name='get_saved_file_path'),

]




