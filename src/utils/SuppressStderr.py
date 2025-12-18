import os
import sys

# 定义一个上下文管理器来屏蔽stderr输出
class SuppressStderr:
    def __enter__(self):
        # 尝试屏蔽stderr，如果失败（例如在某些IDE中）则忽略
        try:
            self.errnull_file = open(os.devnull, 'w')
            self.old_stderr_fileno = os.dup(sys.stderr.fileno())
            os.dup2(self.errnull_file.fileno(), sys.stderr.fileno())
        except Exception:
            self.errnull_file = None
        return self

    def __exit__(self, *_):
        if hasattr(self, 'errnull_file') and self.errnull_file:
            try:
                os.dup2(self.old_stderr_fileno, sys.stderr.fileno())
                os.close(self.old_stderr_fileno)
                self.errnull_file.close()
            except Exception:
                pass
