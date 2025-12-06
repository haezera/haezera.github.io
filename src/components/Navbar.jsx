import { AppBar, Box, Button, Toolbar } from "@mui/material";
import { useNavigate } from "react-router-dom";
import useWindowDimensions from "../helpers/window";
import { useEffect, useState } from "react";
import { gbm } from "../helpers/brownian";
import PriceChart from "./PriceChart";

const Navbar = ({ routes }) => {
  const { width, height } = useWindowDimensions();
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
            hae kim

            <div className="flex flex-row items-center justify-center gap-3">
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