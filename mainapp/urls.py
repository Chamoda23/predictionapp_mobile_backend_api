from django.urls import path
from . import views

urlpatterns = [
    path('text', views.RequestView.as_view(), name='text'),
    path('predict', views.PredictView.as_view(), name='predict')
]