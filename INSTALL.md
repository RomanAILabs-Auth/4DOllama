# Install (quick link)

The **full** install guide lives here:

**[roma4d/docs/Install_Guide.md](roma4d/docs/Install_Guide.md)**

**All Roma4D how-tos** (install, use, LLM, debugging, errors, dependencies, shipping):

**[roma4d/docs/README.md](roma4d/docs/README.md)** · **Single manual:** **[roma4d/docs/Roma4D_Master_Guide.md](roma4d/docs/Roma4D_Master_Guide.md)**

---

## Fast path (repo root)

**Windows:** double-click **`install.cmd`**, or:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\Install-Roma4d.ps1
```

**Linux / macOS:**

```bash
chmod +x scripts/install-roma4d.sh roma4d/install-full.sh
./scripts/install-roma4d.sh
```

Then open a **new** terminal and run **`r4d version`**.
