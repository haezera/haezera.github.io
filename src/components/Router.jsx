import { Box } from "@mui/material";
import { Route, Routes } from "react-router-dom";
import Navbar from "./Navbar";
import Home from "../pages/Home";

const Router = () => {
  const routes = [
    { name: "Home", path: "/" }
  ]

  return (
    <>
      <Box sx={{ 
        display: 'flex', 
        flexDirection: 'column', 
        alignItems: 'center', 
        justifyContent: 'center', 
        height: '100vh'
      }}>
        <Navbar routes={routes}/>
        <Box sx={{ flexGrow: 1, width: "100%", overflow: "auto" }}>
          <Routes>
            <Route path="/" element={<Home />} />
          </Routes>
        </Box>

      </Box>
    </>
  )
};

export default Router;