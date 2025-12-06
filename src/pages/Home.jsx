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
  const sigma = 0.4;
  const dt = 1 / 252;

  useEffect(() => {
    const interval = setInterval(() => {
      setWalk((prevWalk) => {
        const lastPrice = prevWalk[prevWalk.length - 1].p;
        const newPrice = gbmStep(lastPrice, mu, sigma, dt);
        const newWalk = [...prevWalk, { 
          t: prevWalk[prevWalk.length - 1].t + 1, 
          p: newPrice 
        }];

        return newWalk;
      });
    }, 50);

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
        <TextScramble
          texts={["hello, i'm hae"]}
          letterSpeed={40}
          nextLetterSpeed={100}
          pauseTime={1000}
          className="scramble-text"
        />

        <Box width="50%" height="250px" minWidth="400px" sx={{ overflow: 'visible' }}>
        <PriceChart data={walk} />
        </Box>
        <Typography sx={{ mt: 3}}>
          i'm a trader and researcher in the australian quant fund industry
          <br />
          in the past, i've worked on a wide range of problems in financial markets
          <br /> <br/>
          in university, i decided to "open source" my study materials - you can 
          <br />
          find these materials <a style={{ textDecoration: "underline" }} href="/resources"> here</a>.
        </Typography>

      </Box>
    </>
  )
};

export default Home;