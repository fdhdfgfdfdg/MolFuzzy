# Cross-platform validation

Windows and macOS have not been executed for this release. Local Linux results do not substitute for those platforms.

1. Place the complete source tree at the root of the intended GitHub repository.
2. Open Actions, select tests and choose Run workflow. Push and pull-request events also trigger the matrix.
3. Require all six OS/Python jobs to succeed. Python 3.9 on Linux pins NumPy 1.24.0; other jobs record the resolved version.
4. Download each wheel-validation artifact. Retain environment.json, installed-wheel.txt and status.json together with the workflow URL and commit SHA.
5. Confirm the package comes from site-packages, every test passes, CLI JSON passes and returncode is zero. A missing artifact or failed job is not a pass.
6. Only then update the release validation table and manuscript with the observed platform results.

The workflow retains artifacts on failure. Build/setup failures may occur before evidence generation; inspect the Actions job log in that case. Running it requires an accessible GitHub repository with Actions enabled. This source archive does not supply or presume that repository.
