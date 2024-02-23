"""
URL configuration for Financial_Truth_Guard_Web project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
# default present
from django.contrib import admin
from django.urls import path

# add this to import our views file
from Financial_Truth_Guard_Web import views

urlpatterns = [
    path('admin/', admin.site.urls),
    
    # add these to configure our home page (default view) and result web page
    path('', views.home, name='home'),
    path('result/', views.result, name='result'),
]
