import React from 'react';
import ReactDOM from 'react-dom/client';
import './style.css';
import App from './App';
import { BrowserRouter, Route, Routes } from 'react-router-dom';
import { Writings } from './Writings';
import { Projects } from './Projects';
import { Resources } from './Resources';
import { Subjects } from './Subjects';

const root = ReactDOM.createRoot(
  document.getElementById('root') as HTMLElement
);
root.render(
  <BrowserRouter>
    <Routes>
      <Route path="/" element={<App/>}/>
      <Route path="/writings" element ={<Writings/>}/>
      <Route path="/projects" element = {<Projects/>}/>
      <Route path="/resources" element = {<Resources/>}/>
      <Route path="/subjects" element = {<Subjects/>}/>
    </Routes>
  </BrowserRouter> 
);
