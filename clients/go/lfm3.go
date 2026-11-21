// LFM3-1.7B Go SDK
// A Go client library for LFM3-1.7B inference servers
//
// Usage:
//
//	client := lfm3.NewClient("http://localhost:1239")
//	resp, err := client.Chat(context.Background(), "Hello, how are you?", 50)
//	fmt.Println(resp.Choices[0].Message.Content)

package lfm3

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// Client is a Go client for LFM3 inference servers.
type Client struct {
	baseURL    string
	apiKey     string
	httpClient *http.Client
}

// ChatMessage represents a chat message.
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ChatCompletionRequest represents a chat completion request.
type ChatCompletionRequest struct {
	Model     string        `json:"model,omitempty"`
	Messages  []ChatMessage `json:"messages"`
	MaxTokens int           `json:"max_tokens,omitempty"`
	Temperature float64     `json:"temperature,omitempty"`
	TopP      float64     `json:"top_p,omitempty"`
	Stream    bool        `json:"stream,omitempty"`
}

// ChatCompletionChoice represents a single choice in a completion.
type ChatCompletionChoice struct {
	Index        int         `json:"index"`
	Message      ChatMessage `json:"message"`
	FinishReason string      `json:"finish_reason"`
}

// ChatCompletionUsage represents token usage.
type ChatCompletionUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// ChatCompletionResponse represents a chat completion response.
type ChatCompletionResponse struct {
	ID       string                 `json:"id"`
	Object   string                 `json:"object"`
	Created  int64                  `json:"created"`
	Model    string                 `json:"model"`
	Choices  []ChatCompletionChoice `json:"choices"`
	Usage    ChatCompletionUsage    `json:"usage"`
	RequestID string                `json:"request_id,omitempty"`
}

// HealthResponse represents a health check response.
type HealthResponse struct {
	Status      string `json:"status"`
	Provider    string `json:"provider"`
	Model       string `json:"model"`
	Port        int    `json:"port"`
}

// ModelInfo represents model information.
type ModelInfo struct {
	ID       string `json:"id"`
	Object   string `json:"object"`
	Created  int64  `json:"created"`
	OwnedBy  string `json:"owned_by"`
}

// ListModelsResponse represents a list models response.
type ListModelsResponse struct {
	Data []ModelInfo `json:"data"`
}

// NewClient creates a new LFM3 client.
func NewClient(baseURL string) *Client {
	return &Client{
		baseURL:    baseURL,
		httpClient: &http.Client{Timeout: 60 * time.Second},
	}
}

// SetAPIKey sets the API key for authentication.
func (c *Client) SetAPIKey(apiKey string) {
	c.apiKey = apiKey
}

// SetTimeout sets the HTTP client timeout.
func (c *Client) SetTimeout(timeout time.Duration) {
	c.httpClient.Timeout = timeout
}

// Chat sends a simple chat message and returns the response text.
func (c *Client) Chat(ctx context.Context, message string, maxTokens int) (string, error) {
	req := ChatCompletionRequest{
		Model:     "lfm3-1.7b",
		Messages:  []ChatMessage{{Role: "user", Content: message}},
		MaxTokens: maxTokens,
	}
	resp, err := c.ChatCompletion(ctx, req)
	if err != nil {
		return "", err
	}
	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("no choices returned")
	}
	return resp.Choices[0].Message.Content, nil
}

// ChatCompletion sends a chat completion request.
func (c *Client) ChatCompletion(ctx context.Context, req ChatCompletionRequest) (*ChatCompletionResponse, error) {
	return c.doRequest(ctx, "POST", "/v1/chat/completions", req, &ChatCompletionResponse{})
}

// Health checks the server health.
func (c *Client) Health(ctx context.Context) (*HealthResponse, error) {
	return c.doRequest(ctx, "GET", "/health", nil, &HealthResponse{})
}

// ListModels lists available models.
func (c *Client) ListModels(ctx context.Context) (*ListModelsResponse, error) {
	return c.doRequest(ctx, "GET", "/v1/models", nil, &ListModelsResponse{})
}

// doRequest performs an HTTP request and unmarshals the response.
func (c *Client) doRequest(ctx context.Context, method, path string, body interface{}, result interface{}) (*http.Response, error) {
	url := c.baseURL + path

	var bodyReader io.Reader
	if body != nil {
		jsonBody, err := json.Marshal(body)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal request: %w", err)
		}
		bodyReader = bytes.NewReader(jsonBody)
	}

	req, err := http.NewRequestWithContext(ctx, method, url, bodyReader)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	if c.apiKey != "" {
		req.Header.Set("X-API-Key", c.apiKey)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		respBody, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("API error (status %d): %s", resp.StatusCode, string(respBody))
	}

	if result != nil {
		if err := json.NewDecoder(resp.Body).Decode(result); err != nil {
			return resp, fmt.Errorf("failed to decode response: %w", err)
		}
	}

	return resp, nil
}

// Ensure io is imported
var _ = io.Discard
