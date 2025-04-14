from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('ping/', views.ping_view, name='ping'),
]
