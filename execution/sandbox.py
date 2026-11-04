from dataclasses import dataclass, field

@dataclass
class SandboxPolicy:
    image: str = "sci-sandbox:week2"
    cpu_quota: float = 1.0            # cores
    memory_mb: int = 1024             # hard cap
    timeout_s: int = 180              # wall-clock
    pids_limit: int = 128
    network_disabled: bool = True
    read_only_root: bool = True
    drop_caps: list = field(default_factory=lambda: ["ALL"])
    no_new_privileges: bool = True
    tmpfs_size_mb: int = 64
    user: str = "1000:1000"

    def docker_kwargs(self, workdir_mount: str) -> dict:
        return {
            "image": self.image,
            "command": [],                       # ENTRYPOINT runs runner.py
            "detach": False,
            "remove": False,                     # we extract logs first
            "mem_limit": f"{self.memory_mb}m",
            "memswap_limit": f"{self.memory_mb}m",
            "nano_cpus": int(self.cpu_quota * 1e9),
            "pids_limit": self.pids_limit,
            "network_disabled": self.network_disabled,
            "read_only": self.read_only_root,
            "cap_drop": self.drop_caps,
            "security_opt": ["no-new-privileges"] if self.no_new_privileges else [],
            "user": self.user,
            "working_dir": "/workspace",
            "volumes": {workdir_mount: {"bind": "/workspace", "mode": "rw"}},
            "tmpfs": {"/tmp": f"size={self.tmpfs_size_mb}m,noexec,nosuid"},
            "environment": {"PYTHONHASHSEED": "0"},
        }