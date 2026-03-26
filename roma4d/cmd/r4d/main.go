package main

import (
	"os"

	"github.com/RomanAILabs-Auth/Roma4D/internal/cli"
)

func main() {
	os.Exit(cli.Main(os.Args))
}
