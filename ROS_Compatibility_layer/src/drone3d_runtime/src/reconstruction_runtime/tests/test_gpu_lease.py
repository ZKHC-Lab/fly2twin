import os,tempfile,unittest
from pathlib import Path
from drone3d_resources import GpuLease,GpuBusy,require_gpu_available

class LeaseTest(unittest.TestCase):
    def test_busy_rejection_preserves_holder_and_release_allows_next_start(self):
        with tempfile.TemporaryDirectory() as root:
            path=str(Path(root)/'gpu.lock');lease=GpuLease(path)
            try:
                with self.assertRaisesRegex(GpuBusy,'GPU_BUSY'):require_gpu_available(path)
                require_gpu_available(path, allowed_owner_pid=os.getpid())
                with self.assertRaises(GpuBusy):require_gpu_available(path, allowed_owner_pid=os.getpid()+999999)
                os.fstat(lease.fd)
            finally:lease.close()
            require_gpu_available(path)

if __name__=='__main__':unittest.main()
