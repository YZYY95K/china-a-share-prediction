#!/usr/bin/env python3
"""
使用 Kaggle API Token 下载数据
"""

import os
import sys
import subprocess

# 设置 Kaggle API Token (从截图获取)
os.environ['KAGGLE_API_TOKEN'] = 'KGAT_04842b0d8caf09c57ad163d7b180e918'

competition_name = "china-a-share-market-microstructure-prediction"
data_path = "/root/autodl-tmp/"


def install_kaggle():
    """安装 Kaggle CLI"""
    print("检查 Kaggle CLI...")
    try:
        import kaggle
        print("✅ Kaggle 已安装")
        return True
    except ImportError:
        print("正在安装 kaggle...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
        return True


def download_data():
    """下载竞赛数据"""
    print("=" * 70)
    print("开始下载数据")
    print("=" * 70)
    
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)
    
    print(f"\n数据将下载到: {data_path}")
    
    # 使用命令行方式下载
    cmd = [
        "kaggle", "competitions", "download",
        "-c", competition_name,
        "-p", data_path
    ]
    
    print(f"\n执行命令: {' '.join(cmd)}")
    print("\n正在下载（这可能需要一些时间，数据约57GB）...")
    print("=" * 70)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("\n" + "=" * 70)
        print("✅ 下载完成!")
        
        # 列出文件
        print("\n下载的文件:")
        for file in os.listdir(data_path):
            file_path = os.path.join(data_path, file)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path) / (1024 ** 3)
                print(f"  - {file} ({size:.2f} GB)")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 下载失败: {e}")
        return False


def main():
    if not install_kaggle():
        return 1
    
    if download_data():
        print("\n🎉 全部完成!")
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
