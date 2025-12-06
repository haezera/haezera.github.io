import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';
import { useMemo } from 'react';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Legend,
  Filler
);

const lastPointPlugin = {
  id: 'lastPoint',
  afterDraw: (chart) => {
    const ctx = chart.ctx;
    const meta = chart.getDatasetMeta(0);
    if (!meta.data || meta.data.length === 0) return;

    const lastPoint = meta.data[meta.data.length - 1];
    const x = lastPoint.x;
    const y = lastPoint.y;
    const value = chart.data.datasets[0].data[chart.data.datasets[0].data.length - 1];

    ctx.save();
    // Draw white circle
    ctx.fillStyle = '#ffffff';
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(x, y, 2, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = '#ffffff';
    ctx.font = '500 12px Fira Code';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    ctx.fillText(Number(value).toFixed(2), x, y - 8);
    ctx.restore();
  }
};

const PriceChart = ({ data }) => {
  if (!data || data.length === 0) return null;

  const chartData = useMemo(() => {
    const labels = data.map(d => d.t);
    const prices = data.map(d => d.p);

    return {
      labels,
      datasets: [
        {
          label: 'Price',
          data: prices,
          borderColor: '#ffffff',
          backgroundColor: 'transparent',
          borderWidth: 2,
          pointRadius: 0,
          pointHoverRadius: 0,
          tension: 0.4,
          fill: false
        }
      ]
    };
  }, [data]);

  const options = useMemo(() => {
    return {
      responsive: true,
      maintainAspectRatio: false,
      layout: {
        padding: {
          right: 60,
          top: 20,
          bottom: 0,
          left: 0
        }
      },
      plugins: {
        legend: {
          display: false
        },
        tooltip: {
          enabled: false
        }
      },
      scales: {
        x: {
          display: false,
          grid: {
            display: false
          }
        },
        y: {
          display: false,
          grid: {
            display: false
          }
        }
      },
      interaction: {
        intersect: false,
        mode: 'index'
      },
      animation: {
        duration: 0
      },
      elements: {
        point: {
          radius: 0
        }
      }
    };
  }, []);

  return (
    <Line 
      data={chartData} 
      options={options}
      plugins={[lastPointPlugin]}
    />
  );
};

export default PriceChart;