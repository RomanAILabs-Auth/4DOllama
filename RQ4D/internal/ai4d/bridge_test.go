package ai4d

import (
	"testing"
)

func TestParseLineModelLoad(t *testing.T) {
	c, err := ParseLine(`MODEL LOAD "m.4dai"`)
	if err != nil || c.Kind != CmdModelLoad || c.Path != "m.4dai" {
		t.Fatalf("got %+v err=%v", c, err)
	}
}

func TestParseScriptTrainInfer(t *testing.T) {
	src := `
# setup
LAYER CLIFFORD size=4
TRAIN steps=2 lr=0.01
INFER input=[1,0,0,0]
`
	cmds, err := ParseScript(src)
	if err != nil {
		t.Fatal(err)
	}
	if len(cmds) != 3 {
		t.Fatalf("want 3 cmds got %d %+v", len(cmds), cmds)
	}
	if cmds[0].Kind != CmdLayerClifford || cmds[0].Size != 4 {
		t.Fatalf("layer %+v", cmds[0])
	}
	if cmds[1].Steps != 2 || cmds[1].LR != 0.01 {
		t.Fatalf("train %+v", cmds[1])
	}
	if len(cmds[2].Input) != 4 {
		t.Fatalf("infer %+v", cmds[2])
	}
}

func TestParseRotate(t *testing.T) {
	c, err := ParseLine(`ROTATE plane=XY angle=0.785`)
	if err != nil || c.Plane != "XY" || c.Angle < 0.78 {
		t.Fatalf("%+v %v", c, err)
	}
}
