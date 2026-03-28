package compiler

import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
)

// FindZig returns the Zig executable used as `zig cc` on Windows (default native linker).
// Checks R4D_ZIG first (absolute path or PATH name), then PATH.
// On Windows, zig.exe is tried before zig for clarity with WinGet / manual installs.
func FindZig() string {
	if p := strings.TrimSpace(os.Getenv("R4D_ZIG")); p != "" {
		if filepath.IsAbs(p) {
			if st, err := os.Stat(p); err == nil && !st.IsDir() {
				return filepath.Clean(p)
			}
			return ""
		}
		if q, err := exec.LookPath(p); err == nil {
			return q
		}
		if st, err := os.Stat(p); err == nil && !st.IsDir() {
			return filepath.Clean(p)
		}
		return ""
	}
	names := []string{"zig", "zig.exe"}
	if runtime.GOOS == "windows" {
		names = []string{"zig.exe", "zig"}
	}
	for _, name := range names {
		if q, err := exec.LookPath(name); err == nil {
			return q
		}
	}
	return ""
}

// zigWindowsTargetTriple is the `-target` for `zig cc` (MinGW/GNU CRT on Windows).
func zigWindowsTargetTriple() string {
	switch runtime.GOARCH {
	case "arm64":
		return "aarch64-windows-gnu"
	case "386":
		return "x86-windows-gnu"
	default:
		return "x86_64-windows-gnu"
	}
}

// ZigCCCompileArgs returns argv suffix for `zig cc` compile of .ll → .o (for logging and exec).
func ZigCCCompileArgs(llPath, objPath string) []string {
	return []string{"cc", "-target", zigWindowsTargetTriple(), "-c", "-O1", "-o", objPath, llPath}
}

// ZigCCLinkArgs returns argv suffix for `zig cc` link of .o + rt/*.c → exe.
func ZigCCLinkArgs(outExe, objPath string, rtCFiles []string) []string {
	args := []string{"cc", "-target", zigWindowsTargetTriple(), "-o", outExe, objPath}
	return append(args, rtCFiles...)
}

// ZigCompileObjectFromLL builds: zig cc -target <triple> -c -O1 -o objPath llPath
func ZigCompileObjectFromLL(zigExe, llPath, objPath string) *exec.Cmd {
	return exec.Command(zigExe, ZigCCCompileArgs(llPath, objPath)...)
}

// ZigLinkExecutable builds: zig cc -target <triple> -o outExe objPath rtCFiles...
func ZigLinkExecutable(zigExe, outExe, objPath string, rtCFiles []string) *exec.Cmd {
	return exec.Command(zigExe, ZigCCLinkArgs(outExe, objPath, rtCFiles)...)
}
