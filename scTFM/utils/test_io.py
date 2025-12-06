import os
import time
import shutil
import tempfile

def benchmark_io(base_path, size_mb=512, small_file_count=1000):
    """
    测试指定路径的 IO 性能
    :param base_path: 测试目录
    :param size_mb: 大文件测试的大小 (MB)
    :param small_file_count: 小文件测试的数量 (模拟碎片)
    """
    if not os.path.exists(base_path):
        print(f"❌ 路径不存在: {base_path}")
        return

    print(f"\n{'='*60}")
    print(f"🚀 正在测试路径: {base_path}")
    print(f"{'='*60}")

    # 创建临时测试目录
    test_dir = os.path.join(base_path, "io_benchmark_temp")
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)

    try:
        # -------------------------------------------------
        # 1. 大文件顺序写入测试 (Throughput)
        # -------------------------------------------------
        large_file = os.path.join(test_dir, "large_test.dat")
        data_chunk = os.urandom(1024 * 1024) # 1MB chunk
        
        print(f"📦 [1/3] 测试大文件写入 ({size_mb} MB)...")
        start_time = time.time()
        with open(large_file, 'wb') as f:
            for _ in range(size_mb):
                f.write(data_chunk)
            # 强制刷盘，确保不只是写到了内存 Cache 里
            os.fsync(f.fileno())
        
        write_time = time.time() - start_time
        write_speed = size_mb / write_time
        print(f"   ✅ 写入速度: {write_speed:.2f} MB/s (耗时: {write_time:.2f}s)")

        # -------------------------------------------------
        # 2. 大文件顺序读取测试 (Read Throughput)
        # -------------------------------------------------
        print(f"📖 [2/3] 测试大文件读取...")
        # 清除系统缓存 (尝试) - 普通用户权限可能无效，所以这里主要测读取吞吐
        start_time = time.time()
        with open(large_file, 'rb') as f:
            while f.read(1024 * 1024):
                pass
        
        read_time = time.time() - start_time
        read_speed = size_mb / read_time
        print(f"   ✅ 读取速度: {read_speed:.2f} MB/s (耗时: {read_time:.2f}s)")

        # -------------------------------------------------
        # 3. 小文件密集写入测试 (IOPS / Metadata)
        # -------------------------------------------------
        # TileDB 会产生大量小文件，这一步最关键
        print(f"🔨 [3/3] 测试小文件密集创建 ({small_file_count} files)...")
        small_data = b'x' * 4096 # 4KB data
        
        start_time = time.time()
        for i in range(small_file_count):
            fname = os.path.join(test_dir, f"small_{i}.dat")
            with open(fname, 'wb') as f:
                f.write(small_data)
                # 小文件通常依赖 OS 缓存，这里不强制 fsync 以模拟真实应用行为
        
        small_time = time.time() - start_time
        iops = small_file_count / small_time
        print(f"   ✅ 创建速度: {iops:.2f} files/s (耗时: {small_time:.2f}s)")
        
    except Exception as e:
        print(f"❌ 测试出错: {e}")
    finally:
        # 清理
        print(f"🧹 清理测试文件...")
        shutil.rmtree(test_dir)
        print("Done.")

if __name__ == "__main__":
    # 你可以在这里添加你想测试的目录
    paths_to_test = [
        # 1. 你的混合盘 (之前很慢的那个)
        "/gpfs/hybrid/data/jcw", 
        
        # 2. 你的闪存盘 (希望能救命的那个)
        # 请根据你的 df -h 结果，确认你有权限写入 flash 盘的哪个目录
        # 这里假设是你的 home 目录或者你有权限的目录
        "/gpfs/flash/home/jcw", 
        
        # 3. (可选) 如果你有本地盘权限，也可以测测 /tmp
        # "/tmp" 
    ]

    print("开始 IO 性能对比测试...")
    for p in paths_to_test:
        # 检查目录是否可写
        if os.access(p, os.W_OK):
            benchmark_io(p, size_mb=512, small_file_count=2000)
        else:
            print(f"\n❌ 跳过: {p} (路径不存在或无写入权限)")