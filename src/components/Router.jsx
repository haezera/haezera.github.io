import { Box } from "@mui/material";
import { Routes } from "react-router-dom";
import Navbar from "./Navbar";

const Router = () => {
  return (
    <>
      <Box sx={{ 
        display: 'flex', 
        flexDirection: 'column', 
        alignItems: 'center', 
        justifyContent: 'center', 
        height: '100vh'
      }}>
        <Navbar />
        <Box sx={{ flexGrow: 1, width: "100%", overflow: "auto" }}>
          <Routes>

          </Routes>
        </Box>

      </Box>
    </>
  )
};

export default Router;