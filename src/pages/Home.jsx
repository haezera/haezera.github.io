import { Box, Typography } from "@mui/material";
import { TextScramble } from "@skyshots/react-text-scrambler";
import { useEffect, useState } from "react";
import { gbmStep } from "../helpers/brownian";
import PriceChart from "../components/PriceChart";

const Home = () => {
  const [walk, setWalk] = useState([{
    t: 1,
    p: 25
  }]);
  const mu = 0.03;
  const sigma = 1;
  const dt = 1 / 252;

  useEffect(() => {
    const interval = setInterval(() => {
      setWalk((prevWalk) => {
        const lastPrice = prevWalk[prevWalk.length - 1].p;
        const newPrice = gbmStep(lastPrice, mu, sigma, dt);
        return [...prevWalk, { 
          t: prevWalk[prevWalk.length - 1].t + 1, 
          p: newPrice 
        }];
      });
    }, 250);

    return () => clearInterval(interval);
  }, []);

  return (
    <>
      <Box sx={{ 
        display: "flex", 
        flexDirection: "column", 
        justifyContent: "center",
        alignItems: "center",
        width: "100%",
        height: "100%"
      }}>
        <Box width="40%" height="250px">
        <PriceChart data={walk} />
        </Box>
      </Box>
    </>
  )
};

export default Home;