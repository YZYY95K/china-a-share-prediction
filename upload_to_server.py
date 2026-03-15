"""
上传文件到远程服务器
"""
import paramiko
import os

# 服务器配置
HOST = "connect.westc.seetacloud.com"
PORT = 19473
USERNAME = "root"
PASSWORD = "UDNB+OWZrbte"
REMOTE_PATH = "/root/"

# 本地文件
LOCAL_FILE = "train_v28_strict.py"

def upload_file():
    try:
        # 创建SSH客户端
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        print(f"连接服务器 {HOST}:{PORT}...")
        ssh.connect(HOST, port=PORT, username=USERNAME, password=PASSWORD)
        
        # 创建SFTP客户端
        sftp = ssh.open_sftp()
        
        # 上传文件
        local_path = os.path.join(os.path.dirname(__file__), LOCAL_FILE)
        remote_path = REMOTE_PATH + LOCAL_FILE
        
        print(f"上传 {LOCAL_FILE}...")
        sftp.put(local_path, remote_path)
        
        print(f"✅ 上传成功: {remote_path}")
        
        # 检查文件
        stdin, stdout, stderr = ssh.exec_command(f"ls -lh {remote_path}")
        print(f"远程文件信息: {stdout.read().decode()}")
        
        sftp.close()
        ssh.close()
        
    except Exception as e:
        print(f"❌ 错误: {e}")

if __name__ == "__main__":
    upload_file()
