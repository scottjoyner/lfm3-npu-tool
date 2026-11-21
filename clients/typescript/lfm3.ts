// LFM3-1.7B TypeScript SDK
// A TypeScript client library for LFM3-1.7B inference servers
//
// Usage:
//
// import { LFM3Client } from './lfm3';
// const client = new LFM3Client('http://localhost:1239');
// const response = await client.chat('Hello, how are you?', 50);
// console.log(response.choices[0].message.content);

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export interface ChatCompletionRequest {
  model?: string;
  messages: ChatMessage[];
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  stream?: boolean;
}

export interface ChatCompletionChoice {
  index: number;
  message: ChatMessage;
  finish_reason: string;
}

export interface ChatCompletionUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

export interface ChatCompletionResponse {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: ChatCompletionChoice[];
  usage: ChatCompletionUsage;
  request_id?: string;
}

export interface HealthResponse {
  status: string;
  provider: string;
  model: string;
  port: number;
  tls?: Record<string, any>;
}

export interface ModelInfo {
  id: string;
  object: string;
  created: number;
  owned_by: string;
}

export interface ListModelsResponse {
  data: ModelInfo[];
}

export interface LFM3ClientOptions {
  baseURL?: string;
  apiKey?: string;
  timeout?: number;
}

/**
 * LFM3Client is a TypeScript client for LFM3 inference servers.
 */
export class LFM3Client {
  private baseURL: string;
  private apiKey?: string;
  private timeout: number;

  constructor(options: LFM3ClientOptions = {}) {
    this.baseURL = options.baseURL || 'http://localhost:1239';
    this.apiKey = options.apiKey;
    this.timeout = options.timeout || 60000;
  }

  /**
   * Set the API key for authentication.
   */
  setAPIKey(apiKey: string): void {
    this.apiKey = apiKey;
  }

  /**
   * Send a simple chat message and return the response text.
   */
  async chat(message: string, maxTokens: number = 50): Promise<string> {
    const response = await this.chatCompletion({
      model: 'lfm3-1.7b',
      messages: [{ role: 'user', content: message }],
      max_tokens: maxTokens,
    });
    if (response.choices.length === 0) {
      throw new Error('No choices returned');
    }
    return response.choices[0].message.content;
  }

  /**
   * Send a chat completion request.
   */
  async chatCompletion(request: ChatCompletionRequest): Promise<ChatCompletionResponse> {
    return this.request<ChatCompletionResponse>('POST', '/v1/chat/completions', request);
  }

  /**
   * Check server health.
   */
  async health(): Promise<HealthResponse> {
    return this.request<HealthResponse>('GET', '/health');
  }

  /**
   * List available models.
   */
  async listModels(): Promise<ListModelsResponse> {
    return this.request<ListModelsResponse>('GET', '/v1/models');
  }

  /**
   * Get model information.
   */
  async modelInfo(): Promise<ModelInfo> {
    return this.request<ModelInfo>('GET', '/v1/model/info');
  }

  /**
   * Stream chat completion tokens via Server-Sent Events.
   */
  async *chatCompletionStream(
    request: ChatCompletionRequest
  ): AsyncGenerator<string> {
    const url = `${this.baseURL}/v1/chat/completions`;
    const headers: HeadersInit = {
      'Content-Type': 'application/json',
      'Accept': 'text/event-stream',
    };
    if (this.apiKey) {
      headers['X-API-Key'] = this.apiKey;
    }

    const response = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify({ ...request, stream: true }),
      signal: AbortSignal.timeout(this.timeout),
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(`API error (status ${response.status}): ${text}`);
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('No response body');
    }

    const decoder = new TextDecoder();
    let buffer = '';

    try {
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          const trimmed = line.trim();
          if (trimmed.startsWith('data: ')) {
            const data = trimmed.slice(6);
            if (data === '[DONE]') {
              return;
            }
            try {
              const parsed = JSON.parse(data);
              const delta = parsed.choices?.[0]?.delta;
              if (delta?.content) {
                yield delta.content;
              }
            } catch {
              // Skip malformed lines
            }
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  }

  /**
   * Internal HTTP request method.
   */
  private async request<T>(
    method: string,
    path: string,
    body?: any
  ): Promise<T> {
    const url = `${this.baseURL}${path}`;
    const headers: HeadersInit = {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    };
    if (this.apiKey) {
      headers['X-API-Key'] = this.apiKey;
    }

    const response = await fetch(url, {
      method,
      headers,
      body: body ? JSON.stringify(body) : undefined,
      signal: AbortSignal.timeout(this.timeout),
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(`API error (status ${response.status}): ${text}`);
    }

    const contentType = response.headers.get('content-type') || '';
    if (contentType.includes('application/json')) {
      return response.json();
    }
    return response.text() as unknown as T;
  }
}

/**
 * Create a new LFM3 client instance.
 */
export function createClient(options: LFM3ClientOptions = {}): LFM3Client {
  return new LFM3Client(options);
}
