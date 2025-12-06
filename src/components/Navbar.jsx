import { AppBar, Box, Button, Toolbar } from "@mui/material";
import { useNavigate } from "react-router-dom";
import useWindowDimensions from "../helpers/window";
import { useEffect } from "react";
import YahooFinance from "yahoo-finance2";

const yahooFinance = new YahooFinance();

const Navbar = ({ routes }) => {
  const { width, height } = useWindowDimensions();
  const navigate = useNavigate();

  useEffect(() => {
    const fetchPrices = async () => {
      const result = await yahooFinance.quote("AAPL");
      console.log(result);
      return result;
    }
  
    fetchPrices();
  }, []);

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