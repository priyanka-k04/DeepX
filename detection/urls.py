from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static
from django.views.generic import TemplateView

urlpatterns = [
    # --- HTML Template Routes (New) ---
    # These allow you to visit the pages directly on Port 8000
    path('', TemplateView.as_view(template_name='login.html'), name='root'),
    path('upload/', TemplateView.as_view(template_name='upload.html'), name='upload_page'),
    path('results/', TemplateView.as_view(template_name='results.html'), name='results_page'),
    path('history-view/', TemplateView.as_view(template_name='history.html'), name='history_page'),

    # --- Auth API Routes ---
    path('register/', views.register_user, name='register'),
    path('login/', views.login_user, name='login'),
    
    # --- Detection API Routes ---
    path('detect-image/', views.detect_image, name='detect_image'),
    path('detect-video/', views.detect_video, name='detect_video'),
    path('detect-audio/', views.detect_audio, name='detect_audio'),
    
    # --- History API Route ---
    path('history/', views.detection_history, name='detection_history'),
]

# Serve media files (GRAD-CAM heatmaps) during development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)