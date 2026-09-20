# Security policy

Model checkpoints can be an executable serialization format when loaded through unrestricted
Python pickle. This project loads only version-2 checkpoints through PyTorch's
`weights_only=True` restricted loader and does not fall back to unsafe loading.

Treat datasets, calibration files, and checkpoints from unknown sources as untrusted. Verify
published SHA-256 values before use. Report suspected vulnerabilities privately to the
repository owner rather than opening a public issue containing exploit details.
