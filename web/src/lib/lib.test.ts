import { describe, expect, it } from 'vitest'

import { diffChars } from './diff'
import { applyEvent, startRun, visibleOutput } from './run'
import { createSseParser } from './sse'
import { describeViolation, toStep } from './steps'
import type { TransformResult } from './types'

describe('createSseParser', () => {
  it('yields nothing until the blank line that ends an event arrives', () => {
    const parse = createSseParser()
    expect(parse('event: trace\ndata: {"a"')).toEqual([])
    expect(parse(':1}\n')).toEqual([])
    expect(parse('\nevent: token\ndata: {"b":2}\n\n')).toEqual([
      { event: 'trace', data: '{"a":1}' },
      { event: 'token', data: '{"b":2}' },
    ])
  })

  it('handles CRLF and several events in one chunk', () => {
    const parse = createSseParser()
    expect(parse('event: a\r\ndata: 1\r\n\r\nevent: b\r\ndata: 2\r\n\r\n')).toEqual([
      { event: 'a', data: '1' },
      { event: 'b', data: '2' },
    ])
  })

  it('keeps Chinese text and joins multi-line data', () => {
    const parse = createSseParser()
    expect(parse('data: 第一行\ndata: 第二行\n\n')).toEqual([
      { event: 'message', data: '第一行\n第二行' },
    ])
  })
})

describe('diffChars', () => {
  it('marks only what changed', () => {
    expect(diffChars('戊出门了', '戊带着十二块钱出门了')).toEqual([
      { type: 'same', text: '戊' },
      { type: 'added', text: '带着十二块钱' },
      { type: 'same', text: '出门了' },
    ])
  })

  it('reports removals and replacements', () => {
    const segments = diffChars('他老婆问他', '他妻子问他')
    expect(segments.map((item) => item.type)).toEqual(['same', 'removed', 'added', 'same'])
    expect(segments.filter((item) => item.type !== 'removed').map((item) => item.text).join('')).toBe(
      '他妻子问他',
    )
    expect(segments.filter((item) => item.type !== 'added').map((item) => item.text).join('')).toBe(
      '他老婆问他',
    )
  })

  it('returns one segment for identical text and none for two empty strings', () => {
    expect(diffChars('一样', '一样')).toEqual([{ type: 'same', text: '一样' }])
    expect(diffChars('', '')).toEqual([])
  })
})

describe('applyEvent', () => {
  const result: TransformResult = {
    output: '第二版',
    selected_round: 2,
    total_rounds: 2,
    termination: 'accepted',
    fallback_kind: null,
    rounds: [],
    seed: 42,
  }

  it('shows the round being written, then the final text', () => {
    let state = startRun()
    state = applyEvent(state, { event: 'token', data: { round: 1, text: '第一' } })
    state = applyEvent(state, { event: 'token', data: { round: 1, text: '版' } })
    expect(visibleOutput(state)).toBe('第一版')
    state = applyEvent(state, { event: 'token', data: { round: 2, text: '第二' } })
    expect(state.currentRound).toBe(2)
    expect(visibleOutput(state)).toBe('第二')
    expect(state.rounds[1]).toBe('第一版')
    state = applyEvent(state, { event: 'done', data: result })
    expect(state.status).toBe('done')
    expect(visibleOutput(state)).toBe('第二版')
  })

  it('collects trace events in order and records errors', () => {
    let state = startRun()
    for (const node of ['retrieve', 'generate']) {
      state = applyEvent(state, {
        event: 'trace',
        data: { node, ts: '', duration_ms: 1, payload: {} },
      })
    }
    expect(state.trace.map((item) => item.node)).toEqual(['retrieve', 'generate'])
    state = applyEvent(state, { event: 'error', data: { error_type: 'RuntimeError', detail: 'x' } })
    expect(state.status).toBe('error')
    expect(state.error).toBe('RuntimeError')
  })
})

describe('toStep', () => {
  const event = (node: string, payload: Record<string, unknown>) => ({
    node,
    ts: '',
    duration_ms: 12,
    payload,
  })

  it('names the violations a reader can act on', () => {
    const step = toStep(
      event('verify', {
        violation_count: 2,
        violations: [
          { kind: 'numeral_missing', expected: '12块', actual: null },
          { kind: 'title', expected: '太医', actual: '宫监' },
        ],
      }),
    )
    expect(step.tone).toBe('warn')
    expect(step.summary).toBe('发现 2 处事实改动')
    expect(step.details).toEqual(['数值缺失：12块', '称谓改动：太医 → 宫监'])
    expect(describeViolation({ kind: 'entity_hallucination', expected: null, actual: '北京' })).toBe(
      '多出实体：北京',
    )
  })

  it('describes the other nodes', () => {
    expect(toStep(event('retrieve', { exemplar_ids: ['a', 'b'] })).summary).toBe('找到 2 段风格范例')
    expect(toStep(event('retrieve', { exemplar_ids: [] })).summary).toBe('未使用范例')
    expect(toStep(event('verify', { violations: [] })).tone).toBe('ok')
    expect(toStep(event('route', { decision: 'revise' })).summary).toBe('带着修订要求重写')
    const generated = toStep(event('generate', { round: 2, output_chars: 68, echo_stripped_chars: 40 }))
    expect(generated.title).toBe('生成 · 第 2 轮')
    expect(generated.details).toEqual(['切除了复述的修订要求 40 字'])
    expect(toStep(event('generate', { round: 1, raised: true })).tone).toBe('warn')
  })
})
