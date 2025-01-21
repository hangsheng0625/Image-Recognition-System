import computerPic from '../assets/computer.jpg'
export default function Navbar() {
    return (
        <header>
            <nav>
                <img className="nav-logo" src={computerPic} alt="React logo" />
                <span>ReactFacts</span>
            </nav>
        </header>
    )
}