import { AppBar, Box, Button, Toolbar } from "@mui/material";
import { useNavigate } from "react-router-dom";

const Navbar = ({ routes }) => {
  const navigate = useNavigate();

  return (
    <>
      <AppBar 
        position="static" 
        sx={{ 
          width: "100%"
        }}
        color="transparent"
      >
        <Toolbar>
          <div className="flex flex-row items-center justify-between w-full">
            haezera
            <div className="flex flex-row items-center justify-center">
              {
                routes.map((route) => (
                  <Button 
                    onClick={() => navigate(route.path)}
                    sx={{ textDecoration: "none" }}
                  > 
                    {route.name}
                  </Button>
                ))
              }
            </div>
          </div>
        </Toolbar>
      </AppBar>
    </>
  )
};

export default Navbar;