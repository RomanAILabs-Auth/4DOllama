package engine

import (
	"os"
	"strings"
)

// EngineDiagLogsEnabled is true when FOURD_LOG_LEVEL=debug (e.g. after `4dollama serve -verbose`).
// Used to gate stderr diagnostics from the stub/native numeric core during normal chat.
func EngineDiagLogsEnabled() bool {
	return strings.EqualFold(strings.TrimSpace(os.Getenv("FOURD_LOG_LEVEL")), "debug")
}
