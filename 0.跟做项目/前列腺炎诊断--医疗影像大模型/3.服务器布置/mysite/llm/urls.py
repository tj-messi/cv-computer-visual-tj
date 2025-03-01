from django.urls import path
from . import views

urlpatterns = [
    path('', views.toLogin_view, name='index'),
    path('diagnois/', views.diagnois, name='diagnois'),
    path('contact/submit/', views.contact_form_submit, name='contact_submit'),  # 表单提交路由
    path('upload/', views.upload_file, name='upload_file'),  # 文件上传路由
    path('delete/', views.delete_file, name='delete_file'),  # 删除文件路由
]