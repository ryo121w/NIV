from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views  # この行が重要です
from .views import FileUploadView, SpectrumViewSet
from .views import DifferenceGraphView
from .views import SecondDerivativeGraphView
from .views import ThirdDerivativeGraphView
from .views import FourthDerivativeGraphView
from .views import dynamic_graph_view
from .views import SaveMolarAbsorptivityView
from .views import GetSavedFilePathView 
from .views import SecondSaveDerivativeData, DownloadSecondDerivativeData
from .views import ThirdSaveDerivativeData, DownloadThirdDerivativeData
from .views import FourthSaveDerivativeData, DownloadFourthDerivativeData
from .views import DownloadDifferenceData, SaveDifferenceData  # 新しいビューをインポート
from .views import UserCreate
from django.contrib.auth.views import LoginView
from .views import advanced_spectrum_analysis
from .views import gaussian_integral
from django.contrib import admin








router = DefaultRouter()
router.register(r'spectrums', SpectrumViewSet)


urlpatterns = [
    path('api/', include(router.urls)),
    path('api/upload/', FileUploadView.as_view(), name='file-upload'),
    path('api/concentration_graph/', views.ConcentrationGraphView.as_view(), name='concentration-graph'),
    path('api/difference_graph/', DifferenceGraphView.as_view(), name='difference-graph'),
    # 差スペクトルデータを保存するためのエンドポイント
    path('api/save_difference_data/', SaveDifferenceData.as_view(), name='save_difference_data'),
    path('api/download_difference_data/', DownloadDifferenceData.as_view(), name='download_difference_data'),

    path('api/second_derivative_graph/', SecondDerivativeGraphView.as_view(), name='second_derivative_graph'),
    path('api/save_second_derivative_data/', SecondSaveDerivativeData.as_view(), name='save_second_derivative_data'),
    path('api/download_second_derivative_data/', DownloadSecondDerivativeData.as_view(), name='download-second-derivative-data'),


    path('api/third_derivative_graph/', ThirdDerivativeGraphView.as_view(), name='third_derivative_graph'),
    path('api/save_third_derivative_data/', ThirdSaveDerivativeData.as_view(), name='save_third_derivative_data'),
    path('api/download_third_derivative_data/', DownloadThirdDerivativeData.as_view(), name='download_third_derivative_data'),


    path('api/fourth_derivative_graph/', FourthDerivativeGraphView.as_view(), name='fourth_derivative_graph'),
    path('api/save_fourth_derivative_data/', FourthSaveDerivativeData.as_view(), name='save_fourth_derivative_data'),
    path('api/download_fourth_derivative_data/', DownloadFourthDerivativeData.as_view(), name='download_fourth_derivative_data'),



    path('api/dynamic_graph', dynamic_graph_view, name='dynamic_graph'),
    path('api/find_peaks/', views.find_peaks, name='find_peaks'),
    path('api/calculate_hb_strength/', views.calculate_hb_strength, name='calculate_hb_strength'),
    path('api/save_molar_absorptivity/', SaveMolarAbsorptivityView.as_view(), name='save_molar_absorptivity'),
    path('api/get_saved_file_path/', views.GetSavedFilePathView.as_view(), name='get_saved_file_path'),

    #advanced analys
    path('api/advanced/', views.advanced_spectrum_analysis, name='advanced_spectrum_analysis'),
    path('api/gaussian/', views.gaussian_integral),

    path('api/create/', UserCreate.as_view(), name='user-create'),
    path('api/user_login/', LoginView.as_view(), name='login'),

    #LogIn
    path('admin/', admin.site.urls),
    path('accounts/', include('django.contrib.auth.urls')), 


]




