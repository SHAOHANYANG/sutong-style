export interface DiffSegment {
  type: 'same' | 'added' | 'removed'
  text: string
}

// Above this many cells the table is skipped; the texts here are a few hundred characters.
const MAX_CELLS = 4_000_000

/** Character-level diff by longest common subsequence. Chinese has no word boundaries to use. */
export function diffChars(before: string, after: string): DiffSegment[] {
  const a = Array.from(before)
  const b = Array.from(after)
  if (a.length * b.length > MAX_CELLS) {
    const whole: DiffSegment[] = []
    if (before) whole.push({ type: 'removed', text: before })
    if (after) whole.push({ type: 'added', text: after })
    return whole
  }
  const width = b.length + 1
  const table = new Uint32Array((a.length + 1) * width)
  for (let i = a.length - 1; i >= 0; i--) {
    for (let j = b.length - 1; j >= 0; j--) {
      table[i * width + j] =
        a[i] === b[j]
          ? table[(i + 1) * width + j + 1] + 1
          : Math.max(table[(i + 1) * width + j], table[i * width + j + 1])
    }
  }
  const segments: DiffSegment[] = []
  const push = (type: DiffSegment['type'], char: string) => {
    const last = segments[segments.length - 1]
    if (last && last.type === type) last.text += char
    else segments.push({ type, text: char })
  }
  let i = 0
  let j = 0
  while (i < a.length && j < b.length) {
    if (a[i] === b[j]) {
      push('same', a[i])
      i++
      j++
    } else if (table[(i + 1) * width + j] >= table[i * width + j + 1]) {
      push('removed', a[i++])
    } else {
      push('added', b[j++])
    }
  }
  while (i < a.length) push('removed', a[i++])
  while (j < b.length) push('added', b[j++])
  return segments
}
