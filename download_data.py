#!/usr/bin/env python3
"""
下载中国A股市场微观结构预测竞赛数据
从 Kaggle 下载到数据盘
"""

import os
import subprocess
import sys


def install_kaggle():
    """安装 Kaggle CLI"""
    print("安装 Kaggle CLI...")
    try:
        import kaggle
        print("✅ Kaggle 已安装")
        return True
    except ImportError:
        print("正在安装 kaggle...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
        return True


def setup_kaggle_api():
    """设置 Kaggle API 认证"""
    kaggle_dir = os.path.expanduser("~/.kaggle")
    kaggle_json = os.path.join(kaggle_dir, "kaggle.json")

    if not os.path.exists(kaggle_json):
        print("⚠️  未找到 kaggle.json 配置文件")
        print("请按照以下步骤配置 Kaggle API:")
        print("1. 访问 https://www.kaggle.com/settings")
        print("2. 在 'API' 部分点击 'Create New API Token'")
        print("3. 将下载的 kaggle.json 放在 ~/.kaggle/ 目录下")
        print("4. 运行: chmod 600 ~/.kaggle/kaggle.json")
        return False

    # 设置权限
    os.chmod(kaggle_json, 0o600)
    print("✅ Kaggle API 配置已就绪")
    return True


def find_data_disk():
    """查找合适的数据盘"""
    possible_paths = [
        '/root/autodl-tmp/data/',
        '/root/autodl-tmp/',
        '/data/',
        '/dataset/',
        '/mnt/data/',
        './data/',
    ]

    for path in possible_paths:
        if os.path.exists(path) or os.access(os.path.dirname(path), os.W_OK):
            return path

    return './data/'


def download_data():
    """下载竞赛数据"""
    competition_name = "china-a-share-market-microstructure-prediction"
    data_path = find_data_disk()

    print(f"\n数据将下载到: {data_path}")

    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)
        print(f"✅ 创建目录: {data_path}")

    os.chdir(data_path)
    print(f"当前工作目录: {os.getcwd()}")

    # 使用 Kaggle CLI 下载
    print(f"\n开始下载竞赛数据: {competition_name}")
    print("=" * 70)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()

        # 下载竞赛数据
        api.competition_download_files(
            competition=competition_name,
            path=data_path,
            quiet=False,
            force=False
        )

        print("\n" + "=" * 70)
        print("✅ 数据下载完成!")
        print(f"数据位置: {data_path}")

        # 列出下载的文件
        print("\n下载的文件:")
        for file in os.listdir(data_path):
            file_path = os.path.join(data_path, file)
            size = os.path.getsize(file_path) / (1024 ** 3) if os.path.isfile(file_path) else 0
            print(f"  - {file} ({size:.2f} GB)" if size > 0 else f"  - {file}")

        return True

    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        print("\n尝试使用命令行方式...")

        try:
            cmd = [
                "kaggle", "competitions", "download",
                "-c", competition_name,
                "-p", data_path
            ]
            subprocess.check_call(cmd)
            print("✅ 命令行下载成功!")
            return True
        except Exception as e2:
            print(f"❌ 命令行下载也失败: {e2}")
            return False


def main():
    print("=" * 70)
    print("中国A股市场微观结构预测 - 数据下载工具")
    print("=" * 70)

    # 1. 安装 kaggle
    if not install_kaggle():
        return 1

    # 2. 设置 API
    if not setup_kaggle_api():
        print("\n请配置好 kaggle.json 后重新运行此脚本")
        return 1

    # 3. 下载数据
    if download_data():
        print("\n🎉 全部完成!")
        return 0
    else:
        print("\n❌ 下载失败，请检查网络连接或 Kaggle API 配置")
        return 1


if __name__ == "__main__":
    sys.exit(main())
