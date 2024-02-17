import React, { PropsWithChildren } from 'react';

export function Button(props: PropsWithChildren) {
  return (
    <div className="button">
      {props.children}
    </div>
  )
}
