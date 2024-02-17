import React, {useState} from 'react';
import './style.css';
import { Container } from './components/Container';
import { Slides } from './components/Slides'
import { TypeAnimation } from 'react-type-animation';
import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import { Writings } from './Writings';
import { Projects } from './Projects';
import { Subjects } from './Subjects';
import { Resources } from './Resources';
function App() {
  const [isHovered, setIsHovered]: [any, any] = useState(false);
  const [content, setContent]: [any, any] = useState(<h1>Hello, 🌎</h1>);
  const [contentTwo, setContentTwo]: [any, any] = useState(<h2></h2>);
  const [contentThree, setContentThree]: [any, any] = useState(<h2></h2>);
  const [contentFour, setContentFour]: [any, any] = useState(<div className="scroll"/>)
  function dealWithHover() {
    setIsHovered(true);
    setContent(
      <TypeAnimation
        sequence={[
          "I'm Hae."
        ]}
        cursor= {false}
        wrapper="h2"
        speed={50}
      />
    );
    setContentTwo(
      <div>
      <TypeAnimation
        sequence={[
          "I'm a computer science student at UNSW Sydney."
        ]}
        wrapper="h2"
        speed={50}
      />
      </div>
    )
    setContentThree(
      <div>
        <Router>
          <Routes>
            <Route path="/writings" element ={<Writings/>}/>
            <Route path="/projects" element = {<Projects/>}/>
            <Route path="/resources" element = {<Resources/>}/>
            <Route path="/subjects" element = {<Subjects/>}/>
          </Routes>
          <Link className = "button" to ="/writings">writings</Link>
          <Link className = "button" to ="/projects">projects</Link>
          <Link className = "button" to ="/resources">resources</Link>
          <Link className = "button" to ="/subjects">subject reviews</Link>
        </Router> 
      </div>
    );
    setContentFour(<div></div>)
  }
  return (
    <div className="App">
      <Container>
        <Slides>
          <div onMouseOver = {() => {dealWithHover()}}>
            {contentFour}
            {content}
            {contentTwo}
            {contentThree}
          </div>
        </Slides>
      </Container> 
    </div>
  );
}
export default App;
