import React, { PropsWithChildren } from 'react';

export function Project(props: any) {
  return (
  <a href={props.href} style={{textDecoration: "none"}} target="_blank">
  <div className ="project">
    {props.children}
  </div>
  </a>
  )
}
