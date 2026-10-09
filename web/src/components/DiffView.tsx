import { useMemo } from 'react'

import { diffChars } from '../lib/diff'

interface Props {
  before: string
  after: string
}

/** Final text with what the revision added marked, and what it dropped struck through. */
export function DiffView({ before, after }: Props) {
  const segments = useMemo(() => diffChars(before, after), [before, after])
  return (
    <p className="whitespace-pre-wrap break-words font-serif text-lg leading-loose">
      {segments.map((segment, index) => {
        if (segment.type === 'same') return <span key={index}>{segment.text}</span>
        if (segment.type === 'added') {
          return (
            <ins key={index} className="bg-moss/20 text-moss no-underline">
              {segment.text}
            </ins>
          )
        }
        return (
          <del key={index} className="text-cinnabar/80">
            {segment.text}
          </del>
        )
      })}
    </p>
  )
}
