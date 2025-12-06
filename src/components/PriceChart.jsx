// src/components/PriceChart.jsx
import { LineChart, Line, ResponsiveContainer } from "recharts";
import { useMemo } from "react";

const PriceChart = ({ data }) => {
  if (!data || data.length === 0) return null;

  const dataLength = data.length;

  // Memoize the dot component
  const CustomDot = useMemo(() => {
    return ({ cx, cy, index, payload }) => {
      if (index === dataLength - 1) {
        return (
          <g>
            <circle cx={cx} cy={cy} r={4} fill="currentColor" />
            <text
              x={cx}
              y={cy - 12}
              fill="currentColor"
              textAnchor="middle"
              fontSize={12}
              fontWeight="500"
            >
              {payload.p.toFixed(2)}
            </text>
          </g>
        );
      }
      return null;
    };
  }, [dataLength]);

  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={data} margin={{ top: 40, right: 40, bottom: 40, left: 40 }}>
        <Line
          type="monotone"
          dataKey="p"
          dot={CustomDot}
          activeDot={false}
          strokeWidth={2}
          isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
};

export default PriceChart;