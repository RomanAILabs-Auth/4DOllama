package rq4dcore

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
)

// PhysicsTransform fuses lattice diagnostics with a prompt digest (user-space embedding hook).
func PhysicsTransform(norm2 float64, stateSHA256 string, prompt string) string {
	h := sha256.Sum256([]byte(prompt))
	return fmt.Sprintf("rq4d_core|norm2=%.12g|state_sha256=%s|prompt_sha256_prefix=%s",
		norm2, stateSHA256, hex.EncodeToString(h[:8]))
}
