import type { TraceEvent, Violation, ViolationKind } from './types'

export interface Step {
  title: string
  summary: string
  tone: 'neutral' | 'ok' | 'warn'
  details: string[]
  durationMs: number
}

const KIND_LABEL: Record<ViolationKind, string> = {
  entity_missing: '实体缺失',
  numeral_missing: '数值缺失',
  entity_hallucination: '多出实体',
  title: '称谓改动',
}

const DECISION_LABEL: Record<string, string> = {
  accept: '采用这一版',
  revise: '带着修订要求重写',
  re_retrieve: '更换范例后重写',
}

export function describeViolation(item: Violation): string {
  const label = KIND_LABEL[item.kind] ?? item.kind
  if (item.expected && item.actual) return `${label}：${item.expected} → ${item.actual}`
  return `${label}：${item.expected ?? item.actual ?? ''}`
}

/** One row of the trace panel per node visit, in words a reader needs no docs for. */
export function toStep(event: TraceEvent): Step {
  const payload = event.payload
  const base = { durationMs: event.duration_ms, details: [] as string[] }
  switch (event.node) {
    case 'retrieve': {
      const ids = Array.isArray(payload.exemplar_ids) ? (payload.exemplar_ids as string[]) : []
      return {
        ...base,
        title: '检索范例',
        summary: ids.length > 0 ? `找到 ${ids.length} 段风格范例` : '未使用范例',
        tone: 'neutral',
        details: ids,
      }
    }
    case 'generate': {
      const round = Number(payload.round ?? 0)
      const stripped = Number(payload.echo_stripped_chars ?? 0)
      if (payload.raised) {
        return { ...base, title: `生成 · 第 ${round} 轮`, summary: '生成失败', tone: 'warn' }
      }
      return {
        ...base,
        title: `生成 · 第 ${round} 轮`,
        summary: `写出 ${Number(payload.output_chars ?? 0)} 字`,
        tone: 'neutral',
        details: stripped > 0 ? [`切除了复述的修订要求 ${stripped} 字`] : [],
      }
    }
    case 'verify': {
      const violations = Array.isArray(payload.violations) ? (payload.violations as Violation[]) : []
      return {
        ...base,
        title: '保真校验',
        summary: violations.length === 0 ? '通过' : `发现 ${violations.length} 处事实改动`,
        tone: violations.length === 0 ? 'ok' : 'warn',
        details: violations.map(describeViolation),
      }
    }
    case 'score':
      return {
        ...base,
        title: '文体打分',
        summary: typeof payload.score === 'number' ? payload.score.toFixed(3) : '—',
        tone: 'neutral',
      }
    case 'route': {
      const decision = String(payload.decision ?? '')
      return {
        ...base,
        title: '决定',
        summary: DECISION_LABEL[decision] ?? decision,
        tone: decision === 'accept' ? 'ok' : 'neutral',
      }
    }
    default:
      return { ...base, title: event.node, summary: '', tone: 'neutral' }
  }
}
