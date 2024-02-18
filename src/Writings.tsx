import React from 'react';
import { AltColumn } from './components/AltColumn';
import { AltSideColumn } from './components/AltSideColumn';
import { Article } from './components/Article';
import { Menubar } from './components/Menubar';
import { Namecard } from './components/Namecard';
import { Row } from './components/Row';

export function Writings() {
  return (
    <div>
      <Menubar/> <Namecard/>
      <Row>
        <AltSideColumn/>
        <AltColumn>
          <h1>writings (WIP)</h1>
          <Article>
            <h3>Jazz drumming: how it changed my life</h3>
            Jazz drumming is the innate act of swing. 
          </Article>
          <Article>
            <h3>
              Why frontend work makes me want to cry
            </h3>
            Frontend is so sad
          </Article>
        </AltColumn>
        <AltSideColumn/>
      </Row>
    </div>
  )
}
