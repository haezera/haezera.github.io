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
        </AltColumn>
        <AltSideColumn/>
      </Row>
    </div>
  )
}
