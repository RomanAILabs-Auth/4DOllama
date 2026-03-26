// Package tui implements optional full-screen chat (FOURD_TUI=1). Default CLI uses RunLineChat (Ollama-style >>> REPL).
package tui

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/charmbracelet/bubbles/textarea"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/4dollama/4dollama/internal/ollama"
)

const statusLine = "Send a message (/? for help)"

// RunInteractive starts full-screen chat. Uses streaming /api/chat with full message history (Ollama parity).
func RunInteractive(modelName, baseURL string) error {
	m := &chatModel{
		modelName: modelName,
		baseURL:   strings.TrimSuffix(baseURL, "/"),
		messages:  make([]ollama.Message, 0, 32),
	}
	m.initWidgets()
	p := tea.NewProgram(m, tea.WithAltScreen(), tea.WithMouseCellMotion())
	m.send = func(msg tea.Msg) { p.Send(msg) }
	_, err := p.Run()
	return err
}

type chatModel struct {
	modelName string
	baseURL   string
	send      func(tea.Msg)

	viewport  viewport.Model
	textarea  textarea.Model
	width     int
	height    int

	messages  []ollama.Message // full /api/chat history (Ollama parity); raw model text
	history   []string         // plain transcript lines for the viewport (sanitized display)
	streaming string
	busy      bool
	status    string
}

func (m *chatModel) initWidgets() {
	m.textarea = textarea.New()
	m.textarea.Placeholder = statusLine
	m.textarea.ShowLineNumbers = false
	m.textarea.CharLimit = 512000
	m.textarea.SetWidth(72)
	m.textarea.SetHeight(5)
	m.textarea.Focus()
	m.textarea.Prompt = ""

	m.viewport = viewport.New(72, 12)
	m.viewport.SetContent("")
}

func (m *chatModel) Init() tea.Cmd {
	return textarea.Blink
}

func (m *chatModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		m.applyLayout()
		m.viewport, _ = m.viewport.Update(msg)
		m.textarea, _ = m.textarea.Update(msg)
		return m, textarea.Blink

	case tea.MouseMsg:
		var vpCmd tea.Cmd
		m.viewport, vpCmd = m.viewport.Update(msg)
		return m, vpCmd

	case tea.KeyMsg:
		if msg.Type == tea.KeyPgUp {
			m.viewport.LineUp(3)
			return m, nil
		}
		if msg.Type == tea.KeyPgDown {
			m.viewport.LineDown(3)
			return m, nil
		}
		if m.busy {
			if msg.String() == "ctrl+c" {
				return m, tea.Quit
			}
			return m, nil
		}
		switch msg.String() {
		case "ctrl+c", "esc":
			return m, tea.Quit
		case "enter":
			return m.submit()
		case "ctrl+enter", "alt+enter", "f2":
			var cmd tea.Cmd
			m.textarea, cmd = m.textarea.Update(tea.KeyMsg{Type: tea.KeyEnter})
			return m, cmd
		}

	case streamChunkMsg:
		m.streaming += msg.text
		m.refreshViewport()
		return m, nil

	case streamDoneMsg:
		m.busy = false
		if m.streaming != "" {
			m.messages = append(m.messages, ollama.Message{Role: "assistant", Content: m.streaming})
			m.history = append(m.history, sanitizeChatDisplay(m.streaming))
		}
		m.streaming = ""
		m.status = "ready"
		m.refreshViewport()
		return m, textarea.Blink

	case errMsg:
		m.busy = false
		m.history = append(m.history, "error: "+msg.err.Error())
		m.streaming = ""
		m.status = "ready"
		m.refreshViewport()
		return m, textarea.Blink
	}

	var cmd tea.Cmd
	m.textarea, cmd = m.textarea.Update(msg)
	return m, cmd
}

func (m *chatModel) submit() (tea.Model, tea.Cmd) {
	text := strings.TrimSpace(m.textarea.Value())
	if text == "" {
		return m, nil
	}
	low := strings.ToLower(text)
	if low == "/bye" || low == "/exit" || low == "/quit" {
		return m, tea.Quit
	}
	if low == "/help" || low == "/?" {
		help := "Available Commands:\n  /clear          Clear the session context\n  /bye            Exit"
		m.history = append(m.history, help)
		m.textarea.Reset()
		m.refreshViewport()
		return m, textarea.Blink
	}
	if low == "/clear" {
		m.messages = m.messages[:0]
		m.history = m.history[:0]
		m.streaming = ""
		m.viewport.SetContent("")
		m.textarea.Reset()
		m.refreshViewport()
		return m, textarea.Blink
	}

	m.messages = append(m.messages, ollama.Message{Role: "user", Content: text})
	m.history = append(m.history, ">>> "+text)
	m.textarea.Reset()
	m.textarea.Blur()
	m.textarea.Focus()
	m.streaming = ""
	m.busy = true
	m.status = ""
	m.refreshViewport()

	go m.runStreamChat()
	return m, textarea.Blink
}

func (m *chatModel) runStreamChat() {
	url := m.baseURL + "/api/chat"
	tr := true
	body := map[string]any{
		"model":    m.modelName,
		"messages": m.messages,
		"stream":   tr,
	}
	buf, err := json.Marshal(body)
	if err != nil {
		m.send(errMsg{err: err})
		return
	}
	req, err := http.NewRequest(http.MethodPost, url, bytes.NewReader(buf))
	if err != nil {
		m.send(errMsg{err: err})
		return
	}
	req.Header.Set("Content-Type", "application/json")
	client := &http.Client{Timeout: 0}
	resp, err := client.Do(req)
	if err != nil {
		m.send(errMsg{err: fmt.Errorf("%w (is `4dollama serve` running on %s?)", err, m.baseURL)})
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		b, _ := io.ReadAll(resp.Body)
		m.send(errMsg{err: fmt.Errorf("HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(b)))})
		return
	}

	sc := bufio.NewScanner(resp.Body)
	buf2 := make([]byte, 0, 64*1024)
	sc.Buffer(buf2, 4*1024*1024)
	for sc.Scan() {
		var ev struct {
			Message struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"message"`
			Done bool `json:"done"`
		}
		if err := json.Unmarshal(sc.Bytes(), &ev); err != nil {
			continue
		}
		if ev.Message.Content != "" {
			m.send(streamChunkMsg{text: ev.Message.Content})
		}
		if ev.Done {
			break
		}
	}
	if err := sc.Err(); err != nil {
		m.send(errMsg{err: err})
		return
	}
	m.send(streamDoneMsg{})
}

func (m *chatModel) applyLayout() {
	if m.width < 20 {
		return
	}
	w := m.width - 2
	if w < 20 {
		w = m.width
	}
	th := 6
	if m.height > 15 {
		th = 7
	}
	// viewport + gap + textarea (no Ollama-style chrome)
	vh := m.height - th - 3
	if vh < 6 {
		vh = 6
	}
	m.textarea.SetWidth(w)
	m.textarea.SetHeight(th)
	m.viewport.Width = w
	m.viewport.Height = vh
	m.refreshViewport()
}

func (m *chatModel) refreshViewport() {
	var b strings.Builder
	for i, block := range m.history {
		if i > 0 {
			b.WriteByte('\n')
		}
		b.WriteString(block)
	}
	if m.streaming != "" {
		if b.Len() > 0 {
			b.WriteByte('\n')
		}
		b.WriteString(sanitizeChatDisplay(m.streaming))
	}
	m.viewport.SetContent(b.String())
	m.viewport.GotoBottom()
}

func (m *chatModel) View() string {
	body := lipgloss.JoinVertical(lipgloss.Left,
		m.viewport.View(),
		"",
		m.textarea.View(),
	)
	return lipgloss.Place(m.width, m.height, lipgloss.Center, lipgloss.Top, body)
}

type streamChunkMsg struct{ text string }
type streamDoneMsg struct{}
type errMsg struct{ err error }
