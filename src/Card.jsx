import computerPic from './assets/computer.jpg';
function Card (){
    return (
    <div className = "card">

        <img className= "card-image" src={computerPic} alt="computer picture"></img>
        <h2 className="card-title">Liaw Hang Sheng</h2>
        <p className = "card-text">Engineer</p>
    </div>
    )
}

export default Card;