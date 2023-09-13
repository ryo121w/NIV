import glob
from rest_framework import permissions
from rest_framework_simplejwt.views import TokenObtainPairView
from flask import Flask, request, jsonify
import requests
import re
import logging
import uuid
from datetime import datetime
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from django.http import JsonResponse
from django.conf import settings
from django.views import View
from django.http import HttpResponse, JsonResponse, HttpResponseBadRequest, FileResponse
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth import authenticate
from django.contrib.auth.models import User
from rest_framework.authtoken.models import Token
from rest_framework.views import APIView
from rest_framework.parsers import FileUploadParser
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from rest_framework.parsers import FormParser
from rest_framework import status, views
from rest_framework import viewsets
from rest_framework.permissions import AllowAny
from rest_framework.decorators import api_view
from .models import UploadedFile
from .models import Spectrum
from .serializers import SpectrumSerializer
from .serializers import UploadedFileSerializer
from .serializers import UserSerializer
import pandas as pd
from scipy import ndimage
import scipy.signal
from scipy.integrate import simps, trapz
from scipy.signal import find_peaks
from scipy.signal import find_peaks as scipy_find_peaks
from scipy.integrate import quad
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')  # GUIが不要なバックエンド


def my_view(request):
    response = HttpResponse("Here's the text of the Web page.")
    response['Access-Control-Allow-Origin'] = '*'
    return response


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


class SecondSaveDerivativeData(APIView):
    def post(self, request):
        saved_file_path = request.data.get('file_path')
        if not saved_file_path:
            return HttpResponseBadRequest("No saved data found.")

        if not os.path.exists(saved_file_path):
            return HttpResponseBadRequest("No saved data found.")

        df = pd.read_excel(saved_file_path)

        # 二次微分を実行
        columns = df.columns.drop('波長')
        new_df = pd.DataFrame()
        new_df['波長'] = df['波長']

        for col in columns:
            if col.startswith('Molar_Absorptivity_'):
                continue

            smoothed_data = ndimage.gaussian_filter1d(df[col], sigma=10)
            y = ndimage.gaussian_filter1d(smoothed_data, sigma=10, order=2)
            new_df[f'Second_Derivative_{col}'] = y

        # Excelファイルとして保存
        save_path = "/Users/wakiryoutarou/Dropbox/myapp/backend/Second_saved_files/second_derivative_data.xlsx"
        new_df.to_excel(save_path, index=False)  # <--- new_dfを使用

        # ダウンロードURLを返す
        return JsonResponse({"success": True, "download_url": "/api/download_second_derivative_data/"})


class DownloadSecondDerivativeData(APIView):
    def get(self, request):
        file_path = "/Users/wakiryoutarou/Dropbox/myapp/backend/Second_saved_files/second_derivative_data.xlsx"
        if os.path.exists(file_path):
            response = FileResponse(
                open(file_path, 'rb'), content_type='application/vnd.ms-excel')
            response[
                'Content-Disposition'] = f'attachment; filename="{os.path.basename(file_path)}"'
            return response
        else:
            return HttpResponseBadRequest("File not found.")


class ThirdDerivativeGraphView(APIView):
    permission_classes = [AllowAny]

    def post(self, request, *args, **kwargs):
        saved_file_path = request.data.get('file_path')

        if not saved_file_path:
            logger.error("No saved data found.")
            return HttpResponseBadRequest("No saved data found.")

        saved_file_path = os.path.normpath(saved_file_path)

        if not os.path.exists(saved_file_path):
            logger.error("Saved file path does not exist.")
            return HttpResponseBadRequest("No saved data found.")

        df = pd.read_excel(saved_file_path)

        # 三次微分を実行
        columns = df.columns.drop('波長')
        plt.figure(figsize=(10, 6))
        plt.xlim(8000, 6000)
        plt.ylim(-0.000008, 0.000005)

        # カラーマップを設定
        colors = cm.rainbow(np.linspace(0, 1, len(columns)))

        for col, c in zip(columns, colors):
            if col.startswith('Molar_Absorptivity_'):
                continue

            # データの確認
            logger.debug(f"Normalized data for column {col}: {df[col].head()}")

            # 三次微分を行う前にスムージング
            smoothed_data = ndimage.gaussian_filter1d(df[col], sigma=10)

            # 三次微分を行う
            y = ndimage.gaussian_filter1d(smoothed_data, sigma=10, order=3)

            plt.plot(df['波長'], y, label=col, color=c)

        plt.title('Third Derivative of NIR Spectrum')
        plt.xlabel('Wavelength (cm-1)')
        plt.ylabel('Third Derivative of Absorbance')
        plt.legend(loc='upper right')

        # 三次微分されたグラフを保存
        graph_filename = 'third_derivative_nir_spectrum.png'
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


class ThirdSaveDerivativeData(APIView):
    def post(self, request):
        saved_file_path = request.data.get('file_path')
        if not saved_file_path:
            return HttpResponseBadRequest("No saved data found.")

        if not os.path.exists(saved_file_path):
            return HttpResponseBadRequest("No saved data found.")

        df = pd.read_excel(saved_file_path)

        # 三次微分を実行
        columns = df.columns.drop('波長')
        new_df = pd.DataFrame()
        new_df['波長'] = df['波長']

        for col in columns:
            if col.startswith('Molar_Absorptivity_'):
                continue

            smoothed_data = ndimage.gaussian_filter1d(df[col], sigma=10)
            y = ndimage.gaussian_filter1d(smoothed_data, sigma=10, order=3)
            new_df[f'Third_Derivative_{col}'] = y

        # Excelファイルとして保存
        save_path = "/Users/wakiryoutarou/Dropbox/myapp/backend/Third_saved_files/third_derivative_data.xlsx"
        new_df.to_excel(save_path, index=False)

        return JsonResponse({"success": True, "download_url": "/api/download_third_derivative_data/"})


class DownloadThirdDerivativeData(APIView):
    def get(self, request):
        file_path = "/Users/wakiryoutarou/Dropbox/myapp/backend/Third_saved_files/third_derivative_data.xlsx"
        if os.path.exists(file_path):
            response = FileResponse(
                open(file_path, 'rb'), content_type='application/vnd.ms-excel')
            response[
                'Content-Disposition'] = f'attachment; filename="{os.path.basename(file_path)}"'
            return response
        else:
            return HttpResponseBadRequest("File not found.")


class FourthDerivativeGraphView(APIView):
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

        # 四次微分を実行
        columns = df.columns.drop('波長')
        plt.figure(figsize=(10, 6))
        plt.xlim(8000, 6000)
        plt.ylim(-0.0000025, 0.000001)

        # カラーマップを設定
        colors = cm.rainbow(np.linspace(0, 1, len(columns)))

        for col, c in zip(columns, colors):
            if col.startswith('Molar_Absorptivity_'):
                continue

            # 四次微分を行う前にスムージング
            smoothed_data = ndimage.gaussian_filter1d(df[col], sigma=10)

            # 四次微分を行う
            y = ndimage.gaussian_filter1d(smoothed_data, sigma=10, order=4)

            plt.plot(df['波長'], y, label=col, color=c)

        plt.title('Fourth Derivative of NIR Spectrum')
        plt.xlabel('Wavelength (cm-1)')
        plt.ylabel('Fourth Derivative of Absorbance')
        plt.legend(loc='upper right')

        # 四次微分されたグラフを保存
        graph_filename = 'fourth_derivative_nir_spectrum.png'
        graph_dir = 'static'
        graph_filepath = os.path.join(graph_dir, graph_filename)

        if os.path.exists(graph_filepath):
            os.remove(graph_filepath)

        if not os.path.exists(graph_dir):
            os.makedirs(graph_dir)

        plt.savefig(graph_filepath)

        response_data = {'graph_url': os.path.join('static', graph_filename)}
        return JsonResponse(response_data)


class FourthSaveDerivativeData(APIView):
    def post(self, request):
        saved_file_path = request.data.get('file_path')
        if not saved_file_path:
            return HttpResponseBadRequest("No saved data found.")

        if not os.path.exists(saved_file_path):
            return HttpResponseBadRequest("No saved data found.")

        df = pd.read_excel(saved_file_path)

        columns = df.columns.drop('波長')
        new_df = pd.DataFrame()
        new_df['波長'] = df['波長']

        for col in columns:
            if col.startswith('Molar_Absorptivity_'):
                continue

            smoothed_data = ndimage.gaussian_filter1d(df[col], sigma=10)
            y = ndimage.gaussian_filter1d(smoothed_data, sigma=10, order=4)
            new_df[f'Fourth_Derivative_{col}'] = y

        save_path = "/Users/wakiryoutarou/Dropbox/myapp/backend/Fourth_saved_files/fourth_derivative_data.xlsx"
        new_df.to_excel(save_path, index=False)

        return JsonResponse({"success": True, "download_url": "/api/download_fourth_derivative_data/"})


class DownloadFourthDerivativeData(APIView):
    def get(self, request):
        file_path = "/Users/wakiryoutarou/Dropbox/myapp/backend/Fourth_saved_files/fourth_derivative_data.xlsx"
        if os.path.exists(file_path):
            response = FileResponse(
                open(file_path, 'rb'), content_type='application/vnd.ms-excel')
            response[
                'Content-Disposition'] = f'attachment; filename="{os.path.basename(file_path)}"'
            return response
        else:
            return HttpResponseBadRequest("File not found.")


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

            water_concentrations_list = request.data.getlist(
                'concentrations[]', [])

            # Generate a unique filename
            unique_filename = generate_file_id() + ".xlsx"

            # Define the save path
            saved_files_directory = 'saved_files'
            if not os.path.exists(saved_files_directory):
                os.makedirs(saved_files_directory)

            save_path = os.path.join(saved_files_directory, unique_filename)

            df = pd.read_excel(excel_file)
            concentration_columns = [
                col for col in df.columns if re.match(r'\d+M$', col)]
            water_concentrations = {str(molarity): float(concentration) for molarity, concentration in zip(
                concentration_columns, water_concentrations_list)}

            new_save_path = calculate_molar_absorptivity(
                df, water_concentrations, save_path)

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


SAVE_DIR = "/Users/wakiryoutarou/Dropbox/myapp/backend/Difference_saved_files/"


class DifferenceGraphView(APIView):
    parser_classes = [MultiPartParser]

    def get_latest_saved_file_path(self):
        list_of_files = glob.glob('saved_files/*.xlsx')
        latest_saved_file_path = max(
            list_of_files, key=os.path.getctime) if list_of_files else None
        return latest_saved_file_path

    def post(self, request, *args, **kwargs):
        latest_saved_file_path = self.get_latest_saved_file_path()
        if latest_saved_file_path is None:
            return Response({"error": "Could not fetch the latest saved file path"}, status=status.HTTP_400_BAD_REQUEST)

        df = pd.read_excel(latest_saved_file_path)
        zero_m_data = df.get('0M')
        if zero_m_data is None:
            return Response({"error": "0M column not found in the saved file"}, status=status.HTTP_400_BAD_REQUEST)

        columns = [col for col in df.columns if col not in [
            '0M', '波長'] and not col.startswith('Molar_Absorptivity_')]

        for column in columns:
            df[column] -= zero_m_data

            # Baseline Correction using Polynomial Fit
            baseline = np.polyfit(df['波長'], df[column], 3)
            baseline = np.polyval(baseline, df['波長'])
            df[column] -= baseline

        y_min = df.drop(columns=['波長']).min().min()
        y_max = df.drop(columns=['波長']).max().max()

        plt.figure(figsize=(10, 6))
        plt.xlim(8000, 6000)
        plt.ylim(-0.15, 0.1)

        # カラーマップを設定
        colors = cm.rainbow(np.linspace(0, 0.5, len(columns)))

        for column, color in zip(columns, colors):
            plt.plot(df['波長'], df[column], label=column, color=color)

        plt.title('Difference Spectrum with Baseline Correction')
        plt.xlabel('Wavelength (cm-1)')
        plt.ylabel('Difference Intensity')
        plt.legend()

        image_filename = "difference_graph_corrected.png"
        image_path = os.path.join(SAVE_DIR, image_filename)
        plt.savefig(image_path)
        plt.close()

        image_url = f"http://localhost:8000/static/{image_filename}"
        return JsonResponse({"graph_url": image_url})


class SaveDifferenceData(APIView):
    def post(self, request):
        # SAVE_DIRの確認
        # これを実際のディレクトリに置き換えてください
        SAVE_DIR = "/Users/wakiryoutarou/Dropbox/myapp/backend/Difference_saved_files/"
        print(f"SAVE_DIR: {SAVE_DIR}")

        if os.access(SAVE_DIR, os.W_OK):
            print("Write permission is available for SAVE_DIR.")
        else:
            print("Warning: Write permission is NOT available for SAVE_DIR.")

        saved_file_path = request.data.get('file_path')
        if not saved_file_path:
            return HttpResponseBadRequest("No saved data found.")

        if not os.path.exists(saved_file_path):
            return HttpResponseBadRequest("No saved data found.")

        df = pd.read_excel(saved_file_path)
        zero_m_data = df.get('0M')

        if zero_m_data is None:
            return HttpResponseBadRequest("0M column not found in the saved file.")

        columns = [col for col in df.columns if col not in [
            '0M', '波長'] and not col.startswith('Molar_Absorptivity_')]
        new_df = pd.DataFrame()
        new_df['波長'] = df['波長']

        for column in columns:
            df[column] -= zero_m_data

            baseline = np.polyfit(df['波長'], df[column], 3)
            baseline = np.polyval(baseline, df['波長'])
            df[column] -= baseline

            new_df[f'Difference_{column}'] = df[column]

        # new_dfの中身を確認
        print("Content of new_df:")
        print(new_df.head())

        # Excelファイルとして保存
        save_path = os.path.join(SAVE_DIR, "difference_data.xlsx")

        try:
            new_df.to_excel(save_path, index=False)
            print(f"Saved Excel file to {save_path}")
        except Exception as e:
            print(f"Error occurred while saving to Excel: {e}")

        # 単純なテスト
        test_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        test_save_path = os.path.join(SAVE_DIR, "test_data.xlsx")

        try:
            test_df.to_excel(test_save_path, index=False)
            print("Test Excel file has been successfully saved.")
        except Exception as e:
            print(f"Error occurred while saving test Excel file: {e}")

        return JsonResponse({"success": True, "download_url": "/api/download_difference_data/"})


class DownloadDifferenceData(APIView):
    def get(self, request):
        file_path = "/Users/wakiryoutarou/Dropbox/myapp/backend/Difference_saved_files/difference_data.xlsx"
        if os.path.exists(file_path):
            response = FileResponse(
                open(file_path, 'rb'), content_type='application/vnd.ms-excel')
            response[
                'Content-Disposition'] = f'attachment; filename="{os.path.basename(file_path)}"'
            return response
        else:
            return HttpResponseBadRequest("File not found.")


class ConcentrationGraphView(APIView):
    parser_class = (FileUploadParser,)

    def post(self, request):
        print(f"Debug: Received POST data: {request.data}")
        concentrations = request.data.getlist(
            'concentrations[]', [])  # デフォルト値を空のリストとしています。
        # Debug line
        print(f"Debug: Received concentrations: {concentrations}")

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

            colors = cm.rainbow(np.linspace(
                0, 0.5, len(columns)))  # 0.3から1までの範囲で色を設定

            for i, (column, color) in enumerate(zip(columns, colors)):
                df[column] = df[column] / float(concentrations[i])
                plt.plot(df['波長'], df[column],
                         label=f'{column} - {concentrations[i]}M', color=color)

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

            response_data = {'graph_url': os.path.join(
                settings.STATIC_URL, graph_filename)}
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
            colors = cm.rainbow(np.linspace(0, 0.5, len(
                concentrations if concentrations else list(df.columns[1:]))))

            # concentrationsが存在すればそれを使い、なければExcelのカラムを使う
            concentration_columns = concentrations if concentrations else list(
                df.columns[1:])

            for col_name, color in zip(concentration_columns, colors):
                print(f"Debug: col_name = {col_name}, type = {type(col_name)}")
                if col_name in df.columns:
                    plt.plot(df['波長'], df[col_name],
                             label=col_name, color=color)
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


def find_zero_crossings(data):
    return np.where(np.diff(np.sign(data)))[0]


def calculate_area_for_intervals(data, x_data, intervals):
    areas = []
    peak_positions = []
    for start, end in intervals:
        area = simps(data[start:end], x_data[start:end])
        peak_position = x_data[start:end][np.argmax(data[start:end])]
        areas.append(area)
        peak_positions.append(peak_position)
    return areas, peak_positions


@csrf_exempt
def advanced_spectrum_analysis(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['data_file']
        file_extension = uploaded_file.name.split('.')[-1]

        try:
            if file_extension in ['csv']:
                df = pd.read_csv(uploaded_file)
            elif file_extension in ['xlsm', 'xlsx']:
                df = pd.read_excel(uploaded_file, engine='openpyxl')
            else:
                return JsonResponse({'error': 'Unsupported file format'}, status=400)
        except Exception as e:
            return JsonResponse({'error': f'Error reading file: {str(e)}'}, status=400)

        concentrations = list(df.columns)[1:]
        df_filtered = df[(df['波長'] >= 6000) & (df['波長'] <= 8000)]
        results = {}

        for conc in concentrations:
            data = df_filtered[conc].to_numpy()
            x_data = df_filtered['波長'].to_numpy()

            zero_crossings = find_zero_crossings(data)
            intervals = list(
                zip([0] + list(zero_crossings), list(zero_crossings) + [len(data) - 1]))

            areas, peak_positions = calculate_area_for_intervals(
                data, x_data, intervals)

            # 正確なx軸の区間を取得（小数点以下を削除）
            actual_intervals = [(int(x_data[start]), int(x_data[end-1]))
                                for start, end in intervals]

            peak_data = []
            for i, (area, peak_position) in enumerate(zip(areas, peak_positions)):
                peak_data.append({
                    f"area_{chr(120 + i)}": float(area),
                    f"peak_{chr(120 + i)}": float(peak_position),
                })

            results[conc] = {
                'intervals': actual_intervals,
                'areas': [float(area) for area in areas],
                'peak_positions': [float(pos) for pos in peak_positions],
                'peak_data': peak_data
            }

        return JsonResponse({'results': results})

    return JsonResponse({'error': 'Invalid request method'}, status=405)


# データのフィッティング
def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) / 4 / stddev)**2)


def perform_integral(data, lower, upper):
    integral, error = quad(gaussian, lower, upper, args=(
        data['amplitude'], data['mean'], data['stddev']))
    return integral, error


@csrf_exempt
def gaussian_integral(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['data_file']

        # File extension check and reading
        file_extension = uploaded_file.name.split('.')[-1]
        try:
            if file_extension in ['csv']:
                df = pd.read_csv(uploaded_file)
            elif file_extension in ['xlsm', 'xlsx']:
                df = pd.read_excel(uploaded_file, engine='openpyxl')
            else:
                return JsonResponse({'error': 'Unsupported file format'}, status=400)
        except Exception as e:
            return JsonResponse({'error': f'Error reading file: {str(e)}'}, status=400)

        # Filtering data between 6000 and 8000
        df_filtered = df[(df['波長'] >= 6000) & (df['波長'] <= 8000)]

        results = {}
        graph = []

        # Performing Gaussian integral for each concentration
        for column in df_filtered.columns[1:]:
            data = {
                'amplitude': df_filtered[column].max(),
                'mean': df_filtered['波長'][df_filtered[column].idxmax()],
                'stddev': df_filtered[column].std()
            }

            integral, error = perform_integral(data, 6000, 8000)
            results[column] = {'integral': integral, 'error': error}

            # Graph data (this is an example)
            graph.append({
                'x': df_filtered['波長'].tolist(),
                'y': df_filtered[column].tolist(),
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': column
            })

        return JsonResponse({'graph': graph, 'results': results})

    return JsonResponse({'error': 'Invalid request method'}, status=405)


@method_decorator(csrf_exempt, name='dispatch')
class UserCreate(APIView):
    def post(self, request):
        serializer = UserSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class LoginView(TokenObtainPairView):
    permission_classes = (permissions.AllowAny,)

    def post(self, request, *args, **kwargs):
        response = super().post(request, *args, **kwargs)
        if response.status_code == 200:
            user = User.objects.get(username=request.data['username'])
            response.data['user_id'] = user.id
            # Add CORS header manually
            response['Access-Control-Allow-Origin'] = '*'
        return response
