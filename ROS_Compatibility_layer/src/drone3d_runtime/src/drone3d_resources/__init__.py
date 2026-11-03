"""Model-independent process lease. No ROS, CUDA, or reconstruction model imports."""
from contextlib import contextmanager
import fcntl
import os

class GpuBusy(RuntimeError):
    pass

class GpuLease:
    def __init__(self,path=None):
        self.path=path or '/tmp/drone3d-gpu-%d-0.lock'%os.getuid()
        self.fd=os.open(self.path,os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
        try:
            fcntl.flock(self.fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BaseException as error:
            self.close()
            if isinstance(error,BlockingIOError):
                raise GpuBusy('GPU_BUSY: another Drone3D model owns this GPU; it was not interrupted') from error
            raise

    def close(self):
        if self.fd is not None:
            os.close(self.fd);self.fd=None

@contextmanager
def gpu_lease(path=None):
    lease=GpuLease(path)
    try:yield lease
    finally:lease.close()

def require_gpu_available(path=None, allowed_owner_pid=None):
    try:
        with gpu_lease(path): pass
    except GpuBusy:
        if allowed_owner_pid is not None:
            # Only the kernel's live flock owner can justify reusing our own
            # resident worker. A stale PID or an unlocked metadata file cannot.
            target=path or '/tmp/drone3d-gpu-%d-0.lock'%os.getuid()
            stat=os.stat(target,follow_symlinks=False)
            with open('/proc/locks') as source:
                for line in source:
                    fields=line.split()
                    if len(fields)<6 or fields[1]!='FLOCK' or fields[4]!=str(allowed_owner_pid):continue
                    major,minor,inode=fields[5].split(':')
                    if (int(major,16),int(minor,16),int(inode))==(os.major(stat.st_dev),os.minor(stat.st_dev),stat.st_ino):return
        raise
