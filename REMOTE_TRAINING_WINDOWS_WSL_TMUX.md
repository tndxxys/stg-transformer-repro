# Windows 台式机 + WSL + tmux 远程训练（Mac 控制）完整步骤

目标：在 Windows 台式机（有 4070Ti）上跑训练，Mac 通过 SSH 远程控制，不需要切换显示器或键鼠。

这份文档默认你是新手，按顺序操作即可完成。

---

## 0. 准备清单

你需要：
- Windows 台式机（已安装 NVIDIA 驱动）
- Mac（和台式机在同一个局域网）
- 一个可以在 Windows 上登录的用户名和密码

注意：
- 训练会运行在 Windows 台式机上，Mac 只是远程控制。
- tmux 运行在 WSL（Linux）里，因此需要先安装 WSL。

---

## 1. 在 Windows 上安装 WSL2（只需一次）

1) 以管理员身份打开 PowerShell：
   - 右键开始菜单 -> Windows Terminal (管理员) 或 PowerShell (管理员)

2) 执行：
```
wsl --install
```

3) 重启电脑。

4) 打开 Microsoft Store，安装 **Ubuntu 22.04 LTS**（推荐版本）。

5) 启动 Ubuntu，按提示创建 Linux 用户名和密码（记住密码，后面会用）。

---

## 2. 确认 WSL 能看到 GPU（只需一次）

1) 在 Windows 里运行：
```
nvidia-smi
```
能看到 GPU 信息即可。

2) 进入 WSL：
```
wsl
```

3) 在 WSL 里运行：
```
nvidia-smi
```
如果能看到 GPU 信息，说明 WSL GPU 正常。

如果 WSL 里看不到 GPU：
- 更新 NVIDIA 驱动到支持 WSL 的版本（官网最新版即可）。

---

## 3. 在 Windows 上开启 SSH 服务器（只需一次）

1) 在 Windows PowerShell（管理员）执行：
```
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
Start-Service sshd
Set-Service -Name sshd -StartupType 'Automatic'
```

2) 放行防火墙 22 端口：
```
New-NetFirewallRule -Name sshd -DisplayName 'OpenSSH Server (sshd)' -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22
```

---

## 4. 获取 Windows 台式机的 IP 地址

在 Windows PowerShell 执行：
```
ipconfig
```
找到 `IPv4 Address`，记下这个 IP，比如 `192.168.1.100`。172.22.128.1

---

## 5. Mac 通过 SSH 连接 Windows

在 Mac 终端执行：
```
ssh <Windows用户名>@<WindowsIP>
```
示例：
```
ssh yuanzb@172.22.128.121
```

输入 Windows 登录密码即可。

---

## 6. 在 SSH 中进入 WSL

连接成功后执行：
```
wsl
```

看到类似 `username@hostname:~$` 的提示符，说明已进入 WSL。

---

## 7. 在 WSL 中安装 tmux 和基础工具（只需一次）

在 WSL 执行：
```
sudo apt update
sudo apt install -y tmux git python3-pip
```

---

## 8. 把项目放进 WSL（推荐方式）

建议把项目放在 **WSL 的 Linux 目录**，速度更快。

### 方式 A：直接用 git 拉取（推荐）

```
git clone <你的项目仓库地址>
cd <项目目录>
pip install -r requirements.txt
```

### 方式 B：从 Mac 拷贝到 Windows，再移到 WSL

1) 在 Mac 执行（把项目同步到 Windows 用户目录）：
```
rsync -av --delete "/Users/yuanzhibo/Downloads/yzb's programming folder/论文复现/" \
  <Windows用户名>@<WindowsIP>:/Users/<Windows用户名>/project/
```

2) 在 WSL 里把 Windows 目录复制到 Linux 目录：
```
cp -r /mnt/c/Users/<Windows用户名>/project ~/project
cd ~/project
pip install -r requirements.txt
```

---

## 9. 使用 tmux 运行训练（核心步骤）

1) 新建一个 tmux 会话：
```
tmux new -s stg
```

2) 在 tmux 中运行训练命令（示例）：
```
python -m STG_Transformer.train \
  --data_path "UCI CBM Dataset/uci_cbm.csv" \
  --seq_len 96 \
  --pred_len 96 \
  --target_cols "kMc" \
  --drop_cols "date"
```

3) 退出 tmux 但不中断训练：
- 按 `Ctrl+b`，松开后再按 `d`

训练会继续在后台运行。

4) 重新进入 tmux 会话：
```
tmux attach -t stg
```

---

## 10. 检查 GPU 是否在工作

在 WSL 中执行：
```
nvidia-smi
```
如果看到 Python 进程占用 GPU，说明训练已在 GPU 上运行。

---

## 11. 远程查看 TensorBoard（可选）

1) 在 WSL 启动：
```
tensorboard --logdir logs --host 0.0.0.0 --port 6006
```

2) 在 Mac 新终端执行端口转发：
```
ssh -L 6006:localhost:6006 <Windows用户名>@<WindowsIP>
```

3) 在 Mac 浏览器打开：
```
http://localhost:6006
```

---

## 12. 常见问题

### 12.1 SSH 连接不上
- 确认 Windows 防火墙已放行 22 端口
- 确认 Windows IP 正确
- 确认 Windows 用户名/密码正确

### 12.2 WSL 里看不到 GPU
- 更新 NVIDIA 驱动（支持 WSL 的版本）
- 确认 Windows 里 `nvidia-smi` 正常

### 12.3 训练太慢
- 确认是否真的在 GPU 上运行（用 `nvidia-smi` 看）
- 确认你在 WSL 目录下运行，而不是 `/mnt/c` 目录

---

## 13. 快速回顾（最短流程）

1) Windows：装 WSL + Ubuntu  
2) Windows：开 SSH  
3) Mac：`ssh user@ip`  
4) Windows SSH：`wsl`  
5) WSL：`tmux new -s stg`  
6) WSL：跑训练  
7) `Ctrl+b d` 返回  
8) `tmux attach -t stg` 回来查看

---

如果你需要我帮你写一键同步脚本或一键启动训练脚本，告诉我你的 Windows 用户名和项目路径，我可以直接生成。  
