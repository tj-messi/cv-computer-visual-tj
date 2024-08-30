#NVIDIA-smi

---
##install gpu drivers
先ubuntu-drivers list

发现有如下的

nvidia-driver-535-server-open, (kernel modules provided by linux-modules-nvidia-535-server-open-generic-hwe-20.04)

nvidia-driver-535-server, (kernel modules provided by linux-modules-nvidia-535-server-generic-hwe-20.04)

nvidia-driver-535, (kernel modules provided by linux-modules-nvidia-535-generic-hwe-20.04)

nvidia-driver-535-open, (kernel modules provided by linux-modules-nvidia-535-open-generic-hwe-20.04)

再apt install nvidia-driver-535 -y && apt install gdm3 lightdm -y

选择lightdm 然后回车即可

---
##注意
如果出现进不去程序的情况可以执行ctrl+alt+f4进入命令台

然后输入用户名和密码登入系统

进入sudo -i之后

apt remove --purge nvidia*

apt autoremove -y

apt remove --purge gdm3 lightdm -y

把所有安装都清空，然后reboot再安装

apt install nvidia-driver-535 -y

apt install gdm3 lightdm -y

lightdm 回车 再reboot重启即可