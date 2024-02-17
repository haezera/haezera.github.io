import React, { PropsWithChildren } from 'react'
import '../style.css'

export function Slides(props: PropsWithChildren) {
  return (
    <div className = "slides">
      {props.children}
    </div>
  )
}
