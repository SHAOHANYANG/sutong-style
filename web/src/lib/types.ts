export type ViolationKind = 'entity_missing' | 'numeral_missing' | 'entity_hallucination' | 'title'

export interface Violation {
  kind: ViolationKind
  expected: string | null
  actual: string | null
}

export interface TraceEvent {
  node: string
  ts: string
  duration_ms: number
  payload: Record<string, unknown>
}

export interface TokenEvent {
  round: number
  text: string
}

export interface RoundRecord {
  round: number
  output: string
  violation_count: number
  violations: Violation[]
  score: number
  echo_stripped_chars: number
}

export interface TransformResult {
  output: string
  selected_round: number | null
  total_rounds: number
  termination: 'accepted' | 'max_rounds' | 'recursion_limit' | 'fallback'
  fallback_kind: string | null
  rounds: RoundRecord[]
  seed: number
}

export type ServerEvent =
  | { event: 'trace'; data: TraceEvent }
  | { event: 'token'; data: TokenEvent }
  | { event: 'done'; data: TransformResult }
  | { event: 'error'; data: { error_type: string; detail: string } }
