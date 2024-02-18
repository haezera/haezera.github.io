import React from 'react';
import { AltColumn } from './components/AltColumn';
import { AltSideColumn } from './components/AltSideColumn';
import { Menubar } from './components/Menubar';
import { Namecard } from './components/Namecard';
import { Project } from './components/Project';

export function Projects() {
  return (<div>
    <Menubar/> <Namecard/>
    <AltSideColumn/>
    <AltColumn>
      <Project href="https://github.com/haezera/trade">
        <h3>trade: python trading algorithm</h3>
        tech stack: python, flask, pytest, react, jsx <br/> <br/>
        an easy to use cli and (partial) web app that allows you to search
        information about stocks, and use elementary trading strategies on them.
        also has the ability to backtest historical data with these strategies.
      </Project>
      <Project href="https://github.com/haezera/chatroom">
        <h3>chatroom: a simple web socket chatroom</h3>
        tech stack: typescript, express, react, tsx, ws (web sockets), webpack <br/> <br/>
        very elementary web app that allows you to message other users using web sockets.
        peer project with fellow university student.
      </Project>
      <Project href="https://github.com/haezera/haezera.github.io">
        <h3>personal portfolio: the website your on now!</h3>
        tech stack: react, tsx <br/> <br/>
        update from old portfolio which just used html/css. wanted to play with more colors, 
        and a light theme this time around. better interactive design - a nicer home page,
        and overall a lot more happy with this one!
      </Project>
    </AltColumn>
    <AltSideColumn/>
  </div>)
}
