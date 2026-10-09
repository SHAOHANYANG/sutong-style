export interface RawEvent {
  event: string
  data: string
}

/**
 * Incremental parser for a text/event-stream body. EventSource cannot POST, so
 * the stream is read with fetch and fed here chunk by chunk. A chunk may end in
 * the middle of an event; the remainder is kept until the blank line arrives.
 */
export function createSseParser(): (chunk: string) => RawEvent[] {
  let buffer = ''
  return (chunk) => {
    buffer += chunk.replaceAll('\r\n', '\n')
    const events: RawEvent[] = []
    let boundary = buffer.indexOf('\n\n')
    while (boundary !== -1) {
      const block = buffer.slice(0, boundary)
      buffer = buffer.slice(boundary + 2)
      let event = 'message'
      const data: string[] = []
      for (const line of block.split('\n')) {
        if (line.startsWith('event:')) {
          event = line.slice(6).trim()
        } else if (line.startsWith('data:')) {
          const value = line.slice(5)
          data.push(value.startsWith(' ') ? value.slice(1) : value)
        }
      }
      if (data.length > 0) events.push({ event, data: data.join('\n') })
      boundary = buffer.indexOf('\n\n')
    }
    return events
  }
}
