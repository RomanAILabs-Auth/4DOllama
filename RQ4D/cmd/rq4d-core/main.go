// Command rq4d-core runs RQ4D-CORE: daemon, batch, or interactive lattice runtime.
package main

import (
	"os"

	"github.com/RomanAILabs-Auth/RomaQuantum4D/internal/rq4dcore"
)

func main() {
	os.Exit(rq4dcore.Main(os.Args[1:]))
}
