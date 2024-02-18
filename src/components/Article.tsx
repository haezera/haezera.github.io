import React, { PropsWithChildren } from 'react';

export function Article(props: any) {
  return (
    <a href = {props.children} style = {{textDecoration: "none"}}>
    <div className="article">
      {props.children}
    </div>
    </a>
  )
}
