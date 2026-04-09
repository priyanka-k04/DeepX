from django.contrib import admin
from django.urls import path, include
from django.views.generic import TemplateView
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('detection.urls')),
    
    # Mapping all 6 of your pages
    path('', TemplateView.as_view(template_name='index.html'), name='home'),
    path('login/', TemplateView.as_view(template_name='login.html'), name='login'),
    path('register/', TemplateView.as_view(template_name='registration.html'), name='register'),
    path('upload/', TemplateView.as_view(template_name='upload.html'), name='upload'),
    path('results/', TemplateView.as_view(template_name='results.html'), name='results'),
    path('history/', TemplateView.as_view(template_name='history.html'), name='history'),
]

# CRITICAL: This allows your Heatmaps to show up in the results
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)