module github.com/4dollama/4dollama

go 1.22

require (
	github.com/RomanAILabs-Auth/RomaQuantum4D v0.0.0
	github.com/go-chi/chi/v5 v5.2.1
	github.com/spf13/cobra v1.8.1
	golang.org/x/term v0.23.0
)

replace github.com/RomanAILabs-Auth/RomaQuantum4D => ./RQ4D

require (
	github.com/inconshreveable/mousetrap v1.1.0 // indirect
	github.com/spf13/pflag v1.0.5 // indirect
	golang.org/x/sys v0.24.0 // indirect
)
