package types

type ChatMessage struct {
	Role       string        `json:"role"`
	Content    interface{}   `json:"content"`
	ToolCalls  []interface{} `json:"tool_calls,omitempty"`
	Name       interface{}   `json:"name,omitempty"`
	ToolCallId interface{}   `json:"tool_call_id,omitempty"`
}
