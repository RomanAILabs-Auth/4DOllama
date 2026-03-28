package rq4dcore

import (
	"encoding/json"

	"github.com/RomanAILabs-Auth/RomaQuantum4D/internal/ai4d"
)

// TaskFromAICommand maps a parsed Roma4D AI primitive to a scheduler task.
// The caller must set Resp when synchronous completion is required.
// ok is false for CmdNop / unknown commands (avoid colliding with TaskLatticeStep zero value).
func TaskFromAICommand(c ai4d.Command) (t Task, ok bool) {
	switch c.Kind {
	case ai4d.CmdNop:
		return Task{}, false
	case ai4d.CmdModelLoad:
		return Task{Kind: TaskAILoadModel, AIPath: c.Path}, true
	case ai4d.CmdModelSave:
		return Task{Kind: TaskAISaveModel, AIPath: c.Path}, true
	case ai4d.CmdLayerClifford:
		return Task{Kind: TaskAILayerClifford, AICliffN: c.Size}, true
	case ai4d.CmdRotate:
		return Task{Kind: TaskAIRotate4D, AIPlane: c.Plane, AIAngle: c.Angle}, true
	case ai4d.CmdTrain:
		return Task{Kind: TaskAITrainStep, Steps: c.Steps, AILR: c.LR}, true
	case ai4d.CmdInfer:
		b, _ := json.Marshal(c.Input)
		return Task{Kind: TaskAIInference, AIBody: string(b)}, true
	case ai4d.CmdEvolve:
		return Task{Kind: TaskAIEvolveField, AIField: c.Field, Steps: c.Steps}, true
	default:
		return Task{}, false
	}
}
