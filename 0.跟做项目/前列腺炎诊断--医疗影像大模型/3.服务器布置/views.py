from django.shortcuts import render, redirect
from django.core.mail import send_mail
from django.conf import settings
from django.http import JsonResponse
from pathlib import Path
import subprocess
import os,sys



from .utils import get_hello_world_message  # 导入 utils.py 中的函数

# 定义 DATASITE_DIR
DATASITE_DIR = Path(__file__).resolve().parent.parent / 'dataset'


# 首页视图
def toLogin_view(request):
    # 清除上传状态
    if 'file_uploaded' in request.session:
        del request.session['file_uploaded']
    return render(request, 'index.html')


# 诊断页面视图
def diagnois(request):
    # 检查会话中是否有上传状态
    if request.session.get('file_uploaded', False):

        rar_file = '/root/prostate-check-zjz/mysite/dataset/diagnose_source.rar'
        output_dir = '/root/prostate-check-zjz/Medical-SAM2-zjz/data/USVideo_final/test/NoCa'
        subprocess.run(['unrar', 'x', rar_file, output_dir])
        
        folder_count = len([f for f in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, f))])
        old_folder_path = '/root/prostate-check-zjz/Medical-SAM2-zjz/data/USVideo_final/test/NoCa/test'
        new_folder_path = f'/root/prostate-check-zjz/Medical-SAM2-zjz/data/USVideo_final/test/NoCa/{folder_count}'
        os.rename(old_folder_path, new_folder_path)

        new_directory = '/root/prostate-check-zjz/Test-model'
        os.chdir(new_directory)

        python_file = '/root/prostate-check-zjz/Test-model/test-demo.py'
        subprocess.run(['python', python_file])

        txt_path = '/root/prostate-check-zjz/Test-model/Test-sim' + '/' + f'{folder_count}.txt'
        with open(txt_path, 'r') as file:
            lines = file.readlines()
        
        message = 'label:' + lines[-1].strip()
    else:
        message = "暂无文件信息。"
    return render(request, 'diagnois.html', {'message': message})


# 联系我们表单提交视图
def contact_form_submit(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        email = request.POST.get('email')
        message = request.POST.get('message')

        # 邮件内容
        subject = f'New Contact Form Submission from {name}'
        email_message = f'''
        Name: {name}
        Email: {email}
        Message: {message}
        '''

        # 发送邮件
        send_mail(
            subject,
            email_message,
            settings.EMAIL_HOST_USER,  # 发件人
            ['kevinwang363@163.com'],  # 收件人列表
            fail_silently=False,
        )

        # 重定向到首页
        return redirect('index')
    else:
        return redirect('index')


# 文件上传视图
def upload_file(request):
    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']

        # 检查文件是否为 .rar 格式
        if not uploaded_file.name.endswith('.rar'):
            return JsonResponse({'status': 'error', 'message': '请上传RAR格式文件！'}, status=400)

        # 确保 datasite 文件夹存在
        os.makedirs(DATASITE_DIR, exist_ok=True)

        # 固定文件名为 diagnose_source.rar
        save_path = DATASITE_DIR / "diagnose_source.rar"

        # 保存文件到 datasite 文件夹
        with open(save_path, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)

        # 设置会话中的上传状态
        request.session['file_uploaded'] = True

        return JsonResponse({'status': 'success', 'message': '文件上传成功！'})
    else:
        return JsonResponse({'status': 'error', 'message': '无效的请求！'}, status=400)


# 删除文件视图
def delete_file(request):
    # 删除RAR文件
    file_path = DATASITE_DIR / "diagnose_source.rar"
    if file_path.exists():
        os.remove(file_path)

    # 清除上传状态
    if 'file_uploaded' in request.session:
        del request.session['file_uploaded']

    return redirect('diagnois')  # 重定向到诊断页面