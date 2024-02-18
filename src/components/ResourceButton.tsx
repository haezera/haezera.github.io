import React, { PropsWithChildren } from 'react';

export function ResourceButton(props: any) {
  return (
    <a className="resources" href={props.href} target="_blank">{props.children}</a>
  )
}
