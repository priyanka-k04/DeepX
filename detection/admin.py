from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.contrib.auth.models import User
from .models import UserProfile, AnalysisResult

# 1. This makes the Profile (Phone/Full Name) appear inside the User edit page
class UserProfileInline(admin.StackedInline):
    model = UserProfile
    can_delete = False
    verbose_name_plural = 'Extra Profile Info'

# 2. This defines how the User list looks
class UserAdmin(BaseUserAdmin):
    inlines = (UserProfileInline,)
    # Added 'get_full_name' here to show the actual name
    list_display = ('get_full_name', 'email', 'get_phone', 'is_staff')
    
    def get_full_name(self, instance):
        # Pulls the full_name from the UserProfile
        return instance.userprofile.full_name if hasattr(instance, 'userprofile') else instance.username
    get_full_name.short_description = 'Name'

    def get_phone(self, instance):
        return instance.userprofile.phone if hasattr(instance, 'userprofile') else 'No Profile'
    get_phone.short_description = 'Phone Number'

# 3. Re-register the User model
admin.site.unregister(User)
admin.site.register(User, UserAdmin)

# 4. Analysis Result registration
@admin.register(AnalysisResult)
class AnalysisResultAdmin(admin.ModelAdmin):
    list_display = ('file_name', 'media_type', 'prediction', 'confidence', 'timestamp')
    list_filter = ('prediction', 'media_type')
    search_fields = ('file_name',)