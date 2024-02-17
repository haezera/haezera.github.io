import '../style.css'

export function Container(props: any) {
  return (
    <div className = "container">
      {props.children}
    </div>
  )
}
