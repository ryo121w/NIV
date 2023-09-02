from django.http import JsonResponse
from django.conf import settings
from django.views import View
from django.http import HttpResponseBadRequest
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from rest_framework.views import APIView
from rest_framework.parsers import FileUploadParser
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from rest_framework.parsers import FormParser
from rest_framework import status
from rest_framework import viewsets
from .models import UploadedFile
from .models import Spectrum
from .serializers import SpectrumSerializer
from .serializers import UploadedFileSerializer
import pandas as pd
from scipy import ndimage
import scipy.signal
import matplotlib
matplotlib.use('Agg')  # GUIが不要なバックエンド
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from datetime import datetime
import uuid
import logging
import re
import requests
import glob 

class SpectrumViewSet(viewsets.ModelViewSet):
    queryset = Spectrum.objects.all().order_by('wavelength')
    serializer_class = SpectrumSerializer

logger = logging.getLogger(__name__)

class SecondDerivativeGraphView(APIView):
    def post(self, request):
        saved_file_path = request.data.get('file_path')

        if not saved_file_path:
            logger.error("No saved data found.")
            return HttpResponseBadRequest("No saved data found.")

        saved_file_path = os.path.normpath(saved_file_path)

        if not os.path.exists(saved_file_path):
            logger.error("Saved file path does not exist.")
            return HttpResponseBadRequest("No saved data found.")

        df = pd.read_excel(saved_file_path)

        # 二次微分を実行
        columns = df.columns.drop('波長')
        plt.figure(figsize=(10, 6))
        plt.xlim(8000, 6000) 
        plt.ylim(-0.00015, 0.00017)

        # カラーマップを設定
        colors = cm.rainbow(np.linspace(0, 1, len(columns)))

        for col, c in zip(columns, colors):
            if col.startswith('Molar_Absorptivity_'):
                continue 

            # データの確認
            logger.debug(f"Normalized data for column {col}: {df[col].head()}")

            # 二次微分を行う前にスムージング
            smoothed_data = ndimage.gaussian_filter1d(df[col], sigma=10)
            
            # 二次微分を行う
            y = ndimage.gaussian_filter1d(smoothed_data, sigma=10, order=2)
            
            plt.plot(df['波長'], y, label=col, color=c)

        plt.title('Second Derivative of NIR Spectrum')
        plt.xlabel('Wavelength (cm-1)')
        plt.ylabel('Second Derivative of Absorbance')
        plt.legend(loc='upper right')

        # 二次微分されたグラフを保存
        graph_filename = 'second_derivative_nir_spectrum.png'
        graph_dir = 'static'
        graph_filepath = os.path.join(graph_dir, graph_filename)

        # 既存のファイルを削除（もし存在する場合）
        if os.path.exists(graph_filepath):
            os.remove(graph_filepath)

        # ディレクトリが存在しない場合は作成
        if not os.path.exists(graph_dir):
            os.makedirs(graph_dir)

        plt.savefig(graph_filepath)

        # 生成されたグラフのURLをJSONで返す
        response_data = {'graph_url': os.path.join('static', graph_filename)}
        return JsonResponse(response_data)

def get_most_recent_file(directory):
    try:
        files = [os.path.join(directory, f) for f in os.listdir(directory)]
        most_recent_file = max(files, key=os.path.getctime)
        return most_recent_file
    except Exception as e:
        return None

# API View to get the path of the most recently saved file
class GetSavedFilePathView(APIView):
    def get(self, request, *args, **kwargs):
        saved_files_directory = 'saved_files'
        latest_file_path = get_most_recent_file(saved_files_directory)
        if latest_file_path:
            return Response({"file_path": os.path.abspath(latest_file_path)})
        else:
            return Response({"error": "No saved file found"}, status=status.HTTP_404_NOT_FOUND)




logger = logging.getLogger(__name__)

def generate_file_id():
    return str(uuid.uuid4().hex)

class SaveMolarAbsorptivityView(APIView):
    parser_classes = [MultiPartParser]
    
    def post(self, request):
        try:
            excel_file = request.FILES.get('file', None)
            if excel_file is None:
                return Response({"file_saved": False, "error": "Excel file is required."}, status=status.HTTP_400_BAD_REQUEST)
            
            water_concentrations_list = request.data.getlist('concentrations[]', [])
            
            # Generate a unique filename
            unique_filename = generate_file_id() + ".xlsx"
            
            # Define the save path
            saved_files_directory = 'saved_files'
            if not os.path.exists(saved_files_directory):
                os.makedirs(saved_files_directory)
            
            save_path = os.path.join(saved_files_directory, unique_filename)
            
            df = pd.read_excel(excel_file)
            concentration_columns = [col for col in df.columns if re.match(r'\d+M$', col)]
            water_concentrations = {str(molarity): float(concentration) for molarity, concentration in zip(concentration_columns, water_concentrations_list)}
            
            new_save_path = calculate_molar_absorptivity(df, water_concentrations, save_path)
            
            return Response({"file_saved": True, "file_path": new_save_path}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"file_saved": False, "error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

def calculate_molar_absorptivity(df, water_concentrations, save_path):
    for col in df.columns:
        if re.match(r'\d+M$', col):
            molarity = col.replace("M", "")
            water_concentration = water_concentrations.get(molarity, 1)
            new_col_name = f"Molar_Absorptivity_{col}"
            df[new_col_name] = df[col] / water_concentration
    
    df.to_excel(save_path, index=False)
    return save_path
    return new_save_path

class DifferenceGraphView(APIView):
    parser_classes = [MultiPartParser]

    def get_latest_saved_file_path(self):
        list_of_files = glob.glob('saved_files/*.xlsx')  
        latest_saved_file_path = max(list_of_files, key=os.path.getctime) if list_of_files else None
        return latest_saved_file_path

    def post(self, request, *args, **kwargs):
        latest_saved_file_path = self.get_latest_saved_file_path()
        if latest_saved_file_path is None:
            return Response({"error": "Could not fetch the latest saved file path"}, status=status.HTTP_400_BAD_REQUEST)

        df = pd.read_excel(latest_saved_file_path)
        zero_m_data = df.get('0M')
        if zero_m_data is None:
            return Response({"error": "0M column not found in the saved file"}, status=status.HTTP_400_BAD_REQUEST)

        for column in df.columns:
            if column not in ['0M', '波長']:
                df[column] -= zero_m_data

                # Baseline Correction using Polynomial Fit
                baseline = np.polyfit(df['波長'], df[column], 3)
                baseline = np.polyval(baseline, df['波長'])
                df[column] -= baseline

        y_min = df.drop(columns=['波長']).min().min()
        y_max = df.drop(columns=['波長']).max().max()

        plt.figure(figsize=(10, 6))
        plt.xlim(8000, 6000)
        plt.ylim(-0.5, 0.5)
        for column in df.columns:
            if column not in ['0M', '波長']:
                if not column.startswith('Molar_Absorptivity_'):
                    plt.plot(df['波長'], df[column], label=column)

        plt.title('Difference Spectrum with Baseline Correction')
        plt.xlabel('Wavelength (cm-1)')
        plt.ylabel('Difference Intensity')
        plt.legend()

        image_filename = "difference_graph_corrected.png"
        image_path = os.path.join(settings.BASE_DIR, 'static', image_filename)
        plt.savefig(image_path)
        plt.close()

        image_url = f"http://localhost:8000/static/{image_filename}"
        return JsonResponse({"graph_url": image_url})


class ConcentrationGraphView(APIView):
    parser_class = (FileUploadParser,)

    def post(self, request):
        print(f"Debug: Received POST data: {request.data}")  
        concentrations = request.data.getlist('concentrations[]', [])  # デフォルト値を空のリストとしています。
        print(f"Debug: Received concentrations: {concentrations}")   # Debug line

        file_serializer = UploadedFileSerializer(data=request.data)
        
        if file_serializer.is_valid():
            file_serializer.save()
            uploaded_file = file_serializer.validated_data['file']

            df = pd.read_excel(uploaded_file)
            columns = df.columns.drop('波長')
            print(f"Debug: Excel columns: {columns.tolist()}")  # Debug line

            if len(columns) != len(concentrations):
                error_message = f'Mismatch between number of data columns ({len(columns)}) and provided concentrations ({len(concentrations)}). Columns: {columns.tolist()}, Concentrations: {concentrations}'
                return Response({'error': error_message}, status=status.HTTP_400_BAD_REQUEST)

            plt.figure(figsize=(10, 6))
            plt.xlim(8000, 6000)
            plt.ylim(0, 0.03)

            colors = cm.rainbow(np.linspace(0, 0.5, len(columns)))  # 0.3から1までの範囲で色を設定

            for i, (column, color) in enumerate(zip(columns, colors)):
                df[column] = df[column] / float(concentrations[i])
                plt.plot(df['波長'], df[column], label=f'{column} - {concentrations[i]}M', color=color)


            plt.title('NIR Spectrum of LiCl with Concentrations')
            plt.xlabel('Wavelength (cm-1)')
            plt.ylabel('Absorbance')
            plt.legend()

            graph_filename = 'concentration_nir_spectrum.png'
            graph_dir = 'static'
            graph_filepath = os.path.join(graph_dir, graph_filename)

            if not os.path.exists('frontend'):
                os.makedirs('frontend')

            plt.savefig(graph_filepath)
            df.to_excel('frontend/saved_data.xlsx', index=False)

            response_data = {'graph_url': os.path.join(settings.STATIC_URL, graph_filename)}
            return JsonResponse(response_data)
        else:
            return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class FileUploadView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request):
        file_serializer = UploadedFileSerializer(data=request.data)

        if file_serializer.is_valid():
            file_serializer.save()
            uploaded_file = file_serializer.validated_data['file']

            # Concentrationsデータを取得とパース
            concentrations = request.data.get('concentrations')
            if concentrations:
                concentrations = json.loads(concentrations)
                if isinstance(concentrations[0], dict):
                    concentrations = list(concentrations[0].keys())[1:]

            # Excelファイルを読み込む
            df = pd.read_excel(uploaded_file)
            df = df[(df['波長'] >= 6000) & (df['波長'] <= 8000)]

            # グラフ生成
            plt.figure(figsize=(10, 6))
            plt.xlim(6000, 8000)
            plt.ylim(0, 1.6)

            # カラーマップを設定
            colors = cm.rainbow(np.linspace(0, 0.5, len(concentrations if concentrations else list(df.columns[1:]))))

            # concentrationsが存在すればそれを使い、なければExcelのカラムを使う
            concentration_columns = concentrations if concentrations else list(df.columns[1:])

            for col_name, color in zip(concentration_columns, colors):
                print(f"Debug: col_name = {col_name}, type = {type(col_name)}")
                if col_name in df.columns:
                    plt.plot(df['波長'], df[col_name], label=col_name, color=color)
                else:
                    return Response({"error": f"Column {col_name} not found"}, status=status.HTTP_400_BAD_REQUEST)

            plt.title('NIR Spectrum')
            plt.xlabel('Wavelength (cm-1)')
            plt.ylabel('Absorbance')
            plt.legend()

            # PNGファイルとして保存
            graph_filename = 'nir_spectrum.png'
            graph_dir = 'static'
            graph_filepath = os.path.join(graph_dir, graph_filename)

            if not os.path.exists(graph_dir):
                os.makedirs(graph_dir)

            plt.savefig(graph_filepath)
            plt.close()  # リソースの解放

            return Response({'graph_url': f'/static/{graph_filename}'}, status=status.HTTP_200_OK)
        else:
            return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@csrf_exempt
def dynamic_graph_view(request):
    if request.method == "POST":
        # POSTリクエストの場合の処理
        try:
            # 受信したExcelファイルをpandasで読み込む
            excel_file = request.FILES["file"]
            data_frame = pd.read_excel(excel_file)

            # pandasのDataFrameを辞書のリストに変換
            data = data_frame.to_dict(orient="records")

            return JsonResponse(data, safe=False)
        except Exception as e:
            return JsonResponse({"error": str(e)})
    else:
        # GETリクエストの場合の処理（または他のHTTPメソッド）
        return JsonResponse({"message": "Only POST method is allowed"}, status=400)

@csrf_exempt  # CSRF対策を無効化（開発環境でのテスト用）
def find_peaks(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        peaks, _ = scipy.signal.find_peaks(data)
        return JsonResponse({'peaks': peaks.tolist()})
    else:
        return JsonResponse({'error': 'Only POST method is allowed'})

@csrf_exempt
def calculate_hb_strength(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        # ここで水素結合の強度を計算するロジックを書く。
        strength = sum(data) / len(data)  # これは単なるプレースホルダーです
        return JsonResponse({'strength': strength})
    else:
        return JsonResponse({'error': 'Only POST method is allowed'})



















