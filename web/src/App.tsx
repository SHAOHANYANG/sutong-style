import { useCallback, useEffect, useRef, useState } from 'react'

import { DiffView } from './components/DiffView'
import { TracePanel } from './components/TracePanel'
import { checkHealth, streamTransform, type Health } from './lib/api'
import { applyEvent, initialRun, startRun, visibleOutput, type RunState } from './lib/run'

const MAX_CHARS = 600
const SAMPLES = [
  '昨天下午3点，老周一个人去了河边。他在那儿站了两个小时，抽了5根烟，一句话也没说。后来天黑了，他才慢慢走回家。',
  '小梅今年18岁，在镇上的裁缝铺当学徒。老板娘每个月给她30块钱，她攒了一年，想给她妈买一件棉袄。',
  '院子里有一棵石榴树，是爷爷年轻的时候种的。每年秋天结果子的时候，全家人都会回来。',
]

const HEALTH_TEXT: Record<Health, string> = {
  checking: '正在连接推理服务…',
  ready: '推理服务已就绪',
  'not-ready': '推理服务正在启动，模型尚未就绪…',
  unreachable: '暂时连不上推理服务，正在重试…',
}

const TERMINATION_TEXT: Record<string, string> = {
  accepted: '校验通过',
  max_rounds: '达到轮数上限，取违规最少的一版',
  recursion_limit: '触发安全上限',
  fallback: '生成失败，原样返回输入',
}

export default function App() {
  const [text, setText] = useState(SAMPLES[0])
  const [run, setRun] = useState<RunState>(initialRun)
  const [health, setHealth] = useState<Health>('checking')
  const [showDiff, setShowDiff] = useState(true)
  const abort = useRef<AbortController | null>(null)

  // Keep asking until the backend answers: a sleeping host needs time to wake.
  useEffect(() => {
    let cancelled = false
    let timer: ReturnType<typeof setTimeout> | undefined
    const poll = async () => {
      const next = await checkHealth()
      if (cancelled) return
      setHealth(next)
      timer = setTimeout(poll, next === 'ready' ? 30_000 : 4_000)
    }
    void poll()
    return () => {
      cancelled = true
      clearTimeout(timer)
    }
  }, [])

  const submit = useCallback(async () => {
    abort.current?.abort()
    const controller = new AbortController()
    abort.current = controller
    setRun(startRun())
    try {
      await streamTransform(
        text,
        (event) => setRun((state) => applyEvent(state, event)),
        controller.signal,
      )
      setRun((state) =>
        state.status === 'running' ? { ...state, status: 'error', error: '连接中断' } : state,
      )
    } catch (error) {
      if (controller.signal.aborted) return
      const message = error instanceof Error ? error.message : '请求失败'
      setRun((state) => ({ ...state, status: 'error', error: message }))
    }
  }, [text])

  const running = run.status === 'running'
  const trimmed = text.trim()
  const canSubmit = health === 'ready' && !running && trimmed.length > 0 && text.length <= MAX_CHARS
  const output = visibleOutput(run)
  const result = run.result
  const firstRound = result?.rounds[0]?.output
  const revised =
    result !== null && firstRound !== undefined && result.output !== firstRound ? firstRound : null

  return (
    <div className="mx-auto max-w-5xl px-4 py-8 sm:px-6 sm:py-12">
      <header className="mb-8">
        <h1 className="font-serif text-3xl font-semibold tracking-wide">sutong-style</h1>
        <p className="mt-2 max-w-2xl text-sm leading-relaxed text-muted">
          把一段白话改写成苏童的文风。微调后的模型负责写，系统在它外面检索风格范例、核对人名与数字有没有被改动，发现问题就让它重写。下方的面板记录每一步。
        </p>
        <p
          role="status"
          className={`mt-3 inline-flex items-center gap-2 text-xs ${health === 'ready' ? 'text-moss' : 'text-muted'}`}
        >
          <span
            aria-hidden
            className={`h-1.5 w-1.5 rounded-full ${health === 'ready' ? 'bg-moss' : 'animate-pulse bg-muted'}`}
          />
          {HEALTH_TEXT[health]}
        </p>
      </header>

      <main className="space-y-6">
        <div className="grid gap-6 md:grid-cols-2">
          <section className="flex flex-col rounded-lg border border-line bg-panel p-4 sm:p-5">
            <div className="mb-3 flex items-baseline justify-between">
              <h2 className="text-sm font-medium">白话输入</h2>
              <span
                className={`text-xs tabular-nums ${text.length > MAX_CHARS ? 'text-cinnabar' : 'text-muted'}`}
              >
                {text.length} / {MAX_CHARS}
              </span>
            </div>
            <textarea
              value={text}
              onChange={(event) => setText(event.target.value)}
              rows={8}
              aria-label="白话输入"
              className="min-h-48 w-full flex-1 resize-y rounded border border-line bg-transparent p-3 font-serif text-lg leading-loose outline-none focus:border-cinnabar"
            />
            <div className="mt-3 flex flex-wrap items-center gap-2">
              <button
                type="button"
                onClick={() => void submit()}
                disabled={!canSubmit}
                className="rounded bg-cinnabar px-4 py-2 text-sm font-medium text-white transition-opacity disabled:cursor-not-allowed disabled:opacity-40"
              >
                {running ? '改写中…' : '改写'}
              </button>
              {SAMPLES.map((sample, index) => (
                <button
                  key={sample}
                  type="button"
                  onClick={() => setText(sample)}
                  disabled={running}
                  className="rounded border border-line px-3 py-2 text-xs text-muted hover:border-muted disabled:opacity-40"
                >
                  示例 {index + 1}
                </button>
              ))}
            </div>
          </section>

          <section className="flex flex-col rounded-lg border border-line bg-panel p-4 sm:p-5">
            <div className="mb-3 flex flex-wrap items-baseline justify-between gap-2">
              <h2 className="text-sm font-medium">
                改写结果
                {running && run.currentRound > 0 && (
                  <span className="ml-2 font-normal text-muted">第 {run.currentRound} 轮</span>
                )}
              </h2>
              {revised !== null && (
                <label className="flex cursor-pointer items-center gap-1.5 text-xs text-muted">
                  <input
                    type="checkbox"
                    checked={showDiff}
                    onChange={(event) => setShowDiff(event.target.checked)}
                  />
                  标出相对第 1 轮的改动
                </label>
              )}
            </div>
            <div className="min-h-48 flex-1" aria-live="polite">
              {run.status === 'error' ? (
                <p className="text-sm text-cinnabar">出错了：{run.error}。可以稍后再试。</p>
              ) : revised !== null && showDiff && result ? (
                <DiffView before={revised} after={result.output} />
              ) : output ? (
                <p className="whitespace-pre-wrap break-words font-serif text-lg leading-loose">
                  {output}
                </p>
              ) : (
                <p className="text-sm text-muted">{running ? '正在检索范例…' : '结果会显示在这里。'}</p>
              )}
            </div>
            {result && (
              <p className="mt-3 border-t border-line pt-3 text-xs leading-relaxed text-muted">
                共 {result.total_rounds} 轮
                {result.selected_round !== null && `，采用第 ${result.selected_round} 轮`}。
                {TERMINATION_TEXT[result.termination] ?? result.termination}。
              </p>
            )}
          </section>
        </div>

        <section className="rounded-lg border border-line bg-panel p-4 sm:p-5">
          <h2 className="mb-2 text-sm font-medium">执行过程</h2>
          <TracePanel trace={run.trace} running={running} />
        </section>
      </main>

      <footer className="mt-10 text-xs leading-relaxed text-muted">
        <p>
          模型为 Qwen2.5-3B 加 LoRA 微调，仅用于学习与研究。校验基于规则，只能发现人名、数值和称谓的改动，发现不了语义层面的偏差。
        </p>
        <p className="mt-1">
          代码与评估结果：
          <a
            className="underline hover:text-cinnabar"
            href="https://github.com/SHAOHANYANG/sutong-style"
          >
            github.com/SHAOHANYANG/sutong-style
          </a>
        </p>
      </footer>
    </div>
  )
}
