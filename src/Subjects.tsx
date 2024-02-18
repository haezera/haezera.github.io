import React from 'react';
import { AltColumn } from './components/AltColumn';
import { AltSideColumn } from './components/AltSideColumn';
import { Menubar } from './components/Menubar';
import { Namecard } from './components/Namecard';
import { Subject } from './components/Subject';
import { SubjectDropdown } from './components/SubjectDropdown';

export function Subjects() {
  return (<div>
    <Menubar/><Namecard/>
    <AltSideColumn/>
    <AltColumn>
      <h1>subject reviews</h1>
      <Subject>
        <h3>comp1511: programming fundamentals</h3>
        taken: 23t1 <br/>
        mark: 95 <br/>
        difficulty: 5/10 <br/>
        time commitment: 6/10 <br/> <br/>
        thoughts: very well run course. often felt like a huge grind. it was my
        first introduction into programming and is very well linked to the 
        rest of the cse courses. if you work hard during the course, you are almost
        guaranteed a good mark.
      </Subject>
      <Subject>
        <h3>comp1521: computer systems fundamentals</h3>
        taken: 23t3 <br/>
        mark: 88 <br/>
        difficulty: 6/10 <br/>
        time commitment 6/10 <br/> <br/>
        thoughts: well run course. the course is fundamentally fairly easily unti
        week 5 - but once you get files it's fine. once you get to processes
        and threads, it will screw with your brain. i kind of bombed the finals
        relative to my assignments - andrew taylor wrote a fairly hard exam the term
        i took it. the course made me want to take operating systems.
      </Subject>
      <Subject>
        <h3>comp1531: software engineering fundamentals</h3>
        taken: 23t2 <br/>
        mark: 96 <br/>
        difficulty: 2/10 <br/>
        time commitment: <code>rand(1, 10)</code>/10 <br/> <br/>
        thoughts: poorly run course. i've heard it's improved after my term though.
        hayden is a lovely guy, and i love his lecturing style. keeps things abstract,
        and lets us delve into the deeper parts ourselves. i literally soloed the 
        biggest iteration. it was one of the worst times in my life /srs.
      </Subject>
      <Subject>
        <h3>comp2521: data structures and algorithms</h3>
        taken: 23t3 <br/>
        mark: 90 <br/>
        difficulty: 7/10 <br/>
        time commitment: 6/10 <br/> <br/>
        thoughts: decently run course. didn't watch any of kevin luxa's lectures.
        watched hayden's ones way before term started. was proper confused on some
        topics, but that's the charm of the course.
      </Subject>
    </AltColumn>
    <AltSideColumn/>
  </div>)
} 
