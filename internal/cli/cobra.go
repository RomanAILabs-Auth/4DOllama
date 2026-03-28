package cli

import (
	"fmt"
	"log/slog"
	"os"
	"strings"

	"github.com/4dollama/4dollama/internal/version"
	"github.com/spf13/cobra"
)

var (
	rootLog         *slog.Logger
	serveVerbose    bool
	servePort       string
	serveHost       string
	createModelfile string
	convertOutput    string
	convertName      string
	convertNoInstall bool
	convertRunSmoke  bool
)

// Execute runs the Cobra CLI (Ollama-shaped command tree for 4dollama).
func Execute(log *slog.Logger) int {
	rootLog = log
	rootCmd.SetOut(os.Stdout)
	rootCmd.SetErr(os.Stderr)
	rootCmd.CompletionOptions.DisableDefaultCmd = true

	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}
	return 0
}

var rootCmd = &cobra.Command{
	Use:   "4dollama",
	Short: "Drop-in Ollama-compatible CLI and API (RomanAI + silent RQ4D geometry)",
	Long: `4dollama mirrors https://github.com/ollama/ollama (serve, run, chat, pull, list, …).
Completions default to your local Ollama server for natural text; RQ4D lattice work runs in-process on every request without changing the transcript.`,
	Version:       version.Version,
	SilenceErrors: true,
	SilenceUsage:  true,
	Run: func(cmd *cobra.Command, args []string) {
		if len(args) > 0 {
			_, _ = fmt.Fprintf(os.Stderr, "Error: unknown command %q for %q\n\n", args[0], cmd.CommandPath())
		}
		_ = cmd.Help()
		os.Exit(2)
	},
}

func init() {
	rootCmd.AddCommand(
		newServeCmd(),
		newCreateCmd(),
		newShowCmd(),
		newRunCmd(),
		newChatCmd(),
		newStopCmd(),
		newPullCmd(),
		newPushCmd(),
		newSigninCmd(),
		newSignoutCmd(),
		newListCmd(),
		newPsCmd(),
		newCpCmd(),
		newRmCmd(),
		newLaunchCmd(),
		newVersionCmd(),
		newImportOllamaCmd(),
		newBenchmark4dCmd(),
		newBenchmarkCmd(),
		newDoctorCmd(),
		newFourdCmd(),
		newConvertCmd(),
	)
	rootCmd.InitDefaultHelpCmd()
}

func newServeCmd() *cobra.Command {
	c := &cobra.Command{
		Use:   "serve",
		Short: "Start the 4dollama HTTP API (Ollama-compatible routes on FOURD_PORT)",
		Run: func(cmd *cobra.Command, args []string) {
			var rest []string
			if serveVerbose {
				rest = append(rest, "-verbose")
			}
			if servePort != "" {
				rest = append(rest, "-p", servePort)
			}
			if serveHost != "" {
				rest = append(rest, "-h", serveHost)
			}
			os.Exit(cmdServe(rest, rootLog))
		},
	}
	c.Flags().BoolVar(&serveVerbose, "verbose", false, "debug-level logs (4D engine diagnostics on stderr)")
	c.Flags().StringVarP(&servePort, "port", "p", "", "listen port (sets FOURD_PORT)")
	c.Flags().StringVar(&serveHost, "host", "", "listen address (sets FOURD_HOST)")
	return c
}

func newCreateCmd() *cobra.Command {
	c := &cobra.Command{
		Use:   "create MODEL",
		Short: "Create a model from a Modelfile (FROM .gguf or FROM .4dai / sharded JSON romanai merge)",
		Args:  cobra.ExactArgs(1),
		Run: func(cmd *cobra.Command, args []string) {
			os.Exit(CmdCreate(args[0], createModelfile, rootLog))
		},
	}
	c.Flags().StringVarP(&createModelfile, "file", "f", "", "Path to Modelfile (required)")
	_ = c.MarkFlagRequired("file")
	return c
}

func newConvertCmd() *cobra.Command {
	c := &cobra.Command{
		Use:   "convert PATH.gguf [prompt...]",
		Short: "Convert a GGUF checkpoint to native Cl(4,0) F16 .4dai (safetensors), install under FOURD_MODELS, optional run",
		Long: `Reads tensors sequentially (one in RAM at a time), dequantizes common GGML types (F32/F16/BF16, Q4_0/Q4_1/Q5_*/Q8_0, Q4_K),
packs each into 4×4 Clifford blocks as F16, and writes a safetensors-compatible .4dai file.

By default the artifact is registered in FOURD_MODELS and a Modelfile sidecar is written.
Pass --run (-r) to perform a one-shot HTTP completion smoke test (starts local serve if needed).
Use FOURD_INFERENCE=stub unless you intentionally forward to Ollama.`,
		Args: cobra.MinimumNArgs(1),
		Run: func(cmd *cobra.Command, args []string) {
			prompt := strings.TrimSpace(strings.Join(args[1:], " "))
			install := !convertNoInstall
			doRun := convertRunSmoke
			if doRun && !install {
				_, _ = fmt.Fprintln(os.Stderr, "4dollama convert: --run requires install (omit --no-install)")
				os.Exit(2)
			}
			os.Exit(CmdConvert(args[0], convertOutput, convertName, install, doRun, prompt, rootLog))
		},
	}
	c.Flags().StringVarP(&convertOutput, "output", "o", "", "Output .4dai path (default: FOURD_MODELS/<name>.4dai when installing, else <stem>_4d.4dai beside the GGUF)")
	c.Flags().StringVar(&convertName, "name", "", "Model name for install + run (default: derived from GGUF filename)")
	c.Flags().BoolVar(&convertNoInstall, "no-install", false, "Only write the converted file; do not copy into FOURD_MODELS")
	c.Flags().BoolVarP(&convertRunSmoke, "run", "r", false, "After install, call /api/generate for a smoke test (optional trailing words = prompt)")
	return c
}

func newShowCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "show MODEL",
		Short: "Show information for a model (GGUF or romanai .4dai)",
		Args:  cobra.ExactArgs(1),
		Run: func(cmd *cobra.Command, args []string) {
			os.Exit(cmdShow(args, rootLog))
		},
	}
}

func newRunCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "run MODEL [prompt ...]",
		Short: "Run a model (REPL with TTY, or one-shot prompt / stdin)",
		Args:  cobra.MinimumNArgs(1),
		Run: func(cmd *cobra.Command, args []string) {
			os.Exit(cmdRun(args, rootLog))
		},
	}
}

func newChatCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "chat MODEL [prompt ...]",
		Short: "Same as run — interactive or one-shot chat (Ollama parity)",
		Args:  cobra.MinimumNArgs(1),
		Run: func(cmd *cobra.Command, args []string) {
			os.Exit(cmdRun(args, rootLog))
		},
	}
}

func newStopCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "stop MODEL",
		Short: "Stop a running model (API parity; stateless engine is a no-op)",
		Args:  cobra.ExactArgs(1),
		Run: func(cmd *cobra.Command, args []string) {
			os.Exit(CmdStop(args[0], rootLog))
		},
	}
}

func newPullCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "pull MODEL",
		Short: "Pull a GGUF from registry.ollama.ai into FOURD_MODELS (for later R4D transmutation)",
		Args:  cobra.ExactArgs(1),
		Run: func(cmd *cobra.Command, args []string) {
			os.Exit(cmdPull([]string{args[0]}, rootLog))
		},
	}
}

func newPushCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "push MODEL",
		Short: "Push a model to ollama.com (delegates to `ollama` when installed)",
		Args:  cobra.MinimumNArgs(1),
		Run: func(cmd *cobra.Command, args []string) {
			os.Exit(CmdPush(args, rootLog))
		},
	}
}

func newSigninCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "signin",
		Short: "Sign in to ollama.com (delegates to `ollama` when installed)",
		Run: func(cmd *cobra.Command, args []string) {
			os.Exit(CmdSignin(args, rootLog))
		},
	}
}

func newSignoutCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "signout",
		Short: "Sign out from ollama.com (delegates to `ollama` when installed)",
		Run: func(cmd *cobra.Command, args []string) {
			os.Exit(CmdSignout(args, rootLog))
		},
	}
}

func newListCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "list",
		Short: "List models (GGUF + shared Ollama library + romanai .4dai)",
		Run: func(cmd *cobra.Command, args []string) {
			os.Exit(cmdList(nil, rootLog))
		},
	}
}

func newPsCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "ps",
		Short: "List running models",
		Run: func(cmd *cobra.Command, args []string) {
			os.Exit(cmdPs(nil, rootLog))
		},
	}
}

func newCpCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "cp SOURCE DEST",
		Short: "Copy a model tag (preserves .gguf / .4dai extension)",
		Args:  cobra.ExactArgs(2),
		Run: func(cmd *cobra.Command, args []string) {
			os.Exit(cmdCp(args, rootLog))
		},
	}
}

func newRmCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "rm MODEL",
		Short: "Remove a local model file under FOURD_MODELS",
		Args:  cobra.ExactArgs(1),
		Run: func(cmd *cobra.Command, args []string) {
			os.Exit(cmdRm(args, rootLog))
		},
	}
}

func newLaunchCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "launch [url]",
		Short: "Open ollama.com (or URL) in the default browser",
		Run: func(cmd *cobra.Command, args []string) {
			os.Exit(CmdLaunch(args, rootLog))
		},
	}
}

func newVersionCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "version",
		Short: "Print 4dollama version",
		Run: func(cmd *cobra.Command, args []string) {
			fmt.Println(version.Version)
		},
	}
}

func newImportOllamaCmd() *cobra.Command {
	return &cobra.Command{
		Use:    "import-ollama MODEL",
		Short:  "Copy GGUF from local Ollama after `ollama pull`",
		Hidden: true,
		Args:   cobra.ExactArgs(1),
		Run: func(cmd *cobra.Command, args []string) {
			os.Exit(cmdImportOllama(args, rootLog))
		},
	}
}

func newBenchmark4dCmd() *cobra.Command {
	return &cobra.Command{
		Use:    "benchmark-4d",
		Short:  "Compare latency vs OLLAMA_HOST",
		Hidden: true,
		Run: func(cmd *cobra.Command, args []string) {
			os.Exit(cmdBenchmark(nil, rootLog))
		},
	}
}

func newBenchmarkCmd() *cobra.Command {
	return &cobra.Command{
		Use:    "benchmark",
		Short:  "Table: tokens/sec + coherence vs Ollama",
		Hidden: true,
		Run: func(cmd *cobra.Command, args []string) {
			os.Exit(cmdBenchmarkTable(nil, rootLog))
		},
	}
}

func newDoctorCmd() *cobra.Command {
	return &cobra.Command{
		Use:    "doctor",
		Short:  "Health, GPU, models",
		Hidden: true,
		Run: func(cmd *cobra.Command, args []string) {
			os.Exit(cmdDoctor(rootLog))
		},
	}
}

func newFourdCmd() *cobra.Command {
	return &cobra.Command{
		Use:    "fourd [subcommand]",
		Short:  "Native 4D lattice + Cl(4,0) substrate",
		Hidden: true,
		Run: func(cmd *cobra.Command, args []string) {
			os.Exit(cmdFourd(args, rootLog))
		},
	}
}
