import React from 'react';
import { Link } from 'react-router-dom';
import { Subjects } from '../Subjects';
export function Menubar() {
  return (
    <div className="navbar">
      <Link to ="/subjects" className="button">subjects</Link>
      <Link to ="/writings" className="button">writings</Link>
      <Link to ="/resources" className="button">resources</Link>
      <Link to ="/projects" className="button">projects</Link>
    </div>
  )
}
