from django.db import models
from django.contrib.auth.models import User

# ... UserProfile stays the same ...
class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    full_name = models.CharField(max_length=255)
    phone = models.CharField(max_length=15, unique=True)

    def __str__(self):
        return self.full_name
class AnalysisResult(models.Model):
    # --- THE MISSING LINK: Connects this result to a specific user ---
    user = models.ForeignKey(User, related_name='detections', on_delete=models.CASCADE, null=True, blank=True)
    
    # Media info
    file_name = models.CharField(max_length=255)
    media_type = models.CharField(max_length=10, choices=[('image', 'Image'), ('video', 'Video'), ('audio', 'Audio')])
    
    # AI Results
    prediction = models.CharField(max_length=20) 
    confidence = models.CharField(max_length=20) 
    heatmap_url = models.CharField(max_length=500, null=True, blank=True)
    
    # Metadata
    timestamp = models.DateTimeField(auto_now_add=True)

    # Optional: If you want to associate results with a specific user later
    # user_email = models.EmailField(null=True, blank=True)

    def __str__(self):
        return f"{self.user.username if self.user else 'System'} - {self.file_name} ({self.prediction})"

    class Meta:
        ordering = ['-timestamp']