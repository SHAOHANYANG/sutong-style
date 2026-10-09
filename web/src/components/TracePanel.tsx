import { toStep } from '../lib/steps'
import type { TraceEvent } from '../lib/types'

const TONE_CLASS = {
  neutral: 'bg-muted',
  ok: 'bg-moss',
  warn: 'bg-cinnabar',
} as const

interface Props {
  trace: TraceEvent[]
  running: boolean
}

export function TracePanel({ trace, running }: Props) {
  if (trace.length === 0 && !running) {
    return (
      <p className="text-sm text-muted">
        提交一段文字后，这里会逐步显示系统做了什么：找了哪些范例、写了几轮、校验发现了什么、为什么重写。
      </p>
    )
  }
  return (
    <ol className="space-y-0">
      {trace.map((event, index) => {
        const step = toStep(event)
        return (
          <li key={index} className="flex gap-3 border-b border-line py-2.5 last:border-b-0">
            <span
              aria-hidden
              className={`mt-1.5 h-2 w-2 shrink-0 rounded-full ${TONE_CLASS[step.tone]}`}
            />
            <div className="min-w-0 flex-1">
              <div className="flex flex-wrap items-baseline gap-x-3 gap-y-0.5">
                <span className="text-sm font-medium">{step.title}</span>
                <span
                  className={`text-sm ${step.tone === 'warn' ? 'text-cinnabar' : 'text-muted'}`}
                >
                  {step.summary}
                </span>
                <span className="ml-auto text-xs tabular-nums text-muted">
                  {formatDuration(step.durationMs)}
                </span>
              </div>
              {step.details.length > 0 && (
                <ul className="mt-1 space-y-0.5">
                  {step.details.map((detail) => (
                    <li key={detail} className="break-words font-serif text-sm text-muted">
                      {detail}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </li>
        )
      })}
      {running && (
        <li className="flex items-center gap-3 py-2.5 text-sm text-muted">
          <span aria-hidden className="h-2 w-2 shrink-0 animate-pulse rounded-full bg-muted" />
          进行中…
        </li>
      )}
    </ol>
  )
}

function formatDuration(ms: number): string {
  if (ms < 1) return '<1 ms'
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`
}
