import { createSseParser } from './sse'
import type { ServerEvent } from './types'

const configured: string | undefined = import.meta.env.VITE_API_BASE
export const API_BASE = (configured ?? 'http://127.0.0.1:8000').replace(/\/$/, '')

export type Health = 'checking' | 'ready' | 'not-ready' | 'unreachable'

/** 200 means the model server answers; 503 means the API is up but its model is not. */
export async function checkHealth(timeoutMs = 4000): Promise<Health> {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  try {
    const response = await fetch(`${API_BASE}/healthz`, { signal: controller.signal })
    return response.ok ? 'ready' : 'not-ready'
  } catch {
    return 'unreachable'
  } finally {
    clearTimeout(timer)
  }
}

export async function streamTransform(
  text: string,
  onEvent: (event: ServerEvent) => void,
  signal: AbortSignal,
): Promise<void> {
  const response = await fetch(`${API_BASE}/v1/transform`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, stream: true }),
    signal,
  })
  if (!response.ok || !response.body) {
    throw new Error(response.status === 422 ? '输入不符合要求' : `服务返回 ${response.status}`)
  }
  const parse = createSseParser()
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader()
  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    for (const raw of parse(value)) {
      onEvent({ event: raw.event, data: JSON.parse(raw.data) } as ServerEvent)
    }
  }
}
