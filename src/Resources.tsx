import React from 'react';
import { Button } from './components/Button';
import { Column } from './components/Column';
import { Menubar } from './components/Menubar';
import { Namecard } from './components/Namecard';
import { ResourceButton } from './components/ResourceButton';
import { Row } from './components/Row';
import { SideColumn } from './components/SideColumn';

export function Resources() {
  return (<div>
    <Menubar/> <Namecard/>
    <Row>
      <SideColumn/>
      <Column>
        <h1> math1081</h1>
      </Column>
      <Column>
        <h1> lab tests </h1>
        <hr/>
        <ResourceButton href="/1081lt1.pdf">lab test 1</ResourceButton>
        <ResourceButton href="labtest2.pdf">lab test 2</ResourceButton>
      </Column>
      <SideColumn/>
    </Row>
    <Row>
      <SideColumn/>
      <Column>
      </Column>
      <Column>
        <h1> finals </h1>
        <hr/>
        <ResourceButton href="2020 T1_annotated.pdf">2020 t1</ResourceButton>
        <ResourceButton href="2020 T2_annotated.pdf">2020 t2</ResourceButton>
        <ResourceButton href="2020 T3_annotated.pdf">2020 t3</ResourceButton>
        <ResourceButton href="2021 T1_annotated.pdf">2021 t1</ResourceButton>
        <ResourceButton href="2021 T2_annotated.pdf">2021 t2</ResourceButton>
        <ResourceButton href="2021 T3_annotated.pdf">2021 t3</ResourceButton>
        <ResourceButton href="2022 T1_annotated.pdf">2022 t1</ResourceButton>
        <ResourceButton href="2022 T2+_annotated.pdf">2022 t2+</ResourceButton>
      </Column>
      <SideColumn/>
    </Row>
    <Row>
      <SideColumn/>
      <Column>
        <h1> math1231</h1>
      </Column>
      <Column>
        <h1> finals </h1>
        <hr/>
        <ResourceButton href="/2020 T1.pdf">2020 t1</ResourceButton>
        <ResourceButton href="/2020 T2.pdf">2020 t2</ResourceButton>
        <ResourceButton href="/2020 T3.pdf">2020 t3</ResourceButton>
        <ResourceButton href="/2021 T1.pdf">2021 t1</ResourceButton>
        <ResourceButton href="/2021 T2.pdf">2021 t2</ResourceButton>
        <ResourceButton href="/2021 T3.pdf">2021 t3</ResourceButton>
        <ResourceButton href="/2022 T1.pdf">2022 t1</ResourceButton>
        <ResourceButton href="/2022 T2 and T3.pdf">2022 t2 & t3</ResourceButton>
        <ResourceButton href="/2023 T1.pdf">2023 t1</ResourceButton>
      </Column>
      <SideColumn/>
    </Row>
    <Row>
      <SideColumn/>
      <Column>
        <h1> comp3121 </h1>
      </Column>
      <Column>
        <ResourceButton href="/COMP3121 Exam Notes.pdf">truncated course notes (exam)</ResourceButton>
        <ResourceButton href="/3121_extensive_notes.pdf">extensive course notes</ResourceButton>
        <ResourceButton href="/Lecture Examples.pdf">lecture examples summary</ResourceButton>
      </Column>
      <SideColumn/>
    </Row>
  </div>)
}
