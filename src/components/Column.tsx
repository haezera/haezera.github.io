import React, { PropsWithChildren } from 'react';

export function Column(props: PropsWithChildren) {
  return (
    <div className="column">
      {props.children}
    </div>
  )
}
