import type { ServerEvent, TraceEvent, TransformResult } from './types'

export interface RunState {
  status: 'idle' | 'running' | 'done' | 'error'
  trace: TraceEvent[]
  /** Text streamed so far, by generation round. */
  rounds: Record<number, string>
  currentRound: number
  result: TransformResult | null
  error: string | null
}

export const initialRun: RunState = {
  status: 'idle',
  trace: [],
  rounds: {},
  currentRound: 0,
  result: null,
  error: null,
}

export function startRun(): RunState {
  return { ...initialRun, status: 'running' }
}

/** Fold one server event into the page state. Pure, so it is tested without a browser. */
export function applyEvent(state: RunState, incoming: ServerEvent): RunState {
  switch (incoming.event) {
    case 'trace':
      return { ...state, trace: [...state.trace, incoming.data] }
    case 'token': {
      const { round, text } = incoming.data
      return {
        ...state,
        currentRound: Math.max(state.currentRound, round),
        rounds: { ...state.rounds, [round]: (state.rounds[round] ?? '') + text },
      }
    }
    case 'done':
      return { ...state, status: 'done', result: incoming.data }
    case 'error':
      return { ...state, status: 'error', error: incoming.data.error_type }
  }
}

/** What the output pane shows: the final text once known, otherwise the round being written. */
export function visibleOutput(state: RunState): string {
  if (state.result) return state.result.output
  return state.rounds[state.currentRound] ?? ''
}
