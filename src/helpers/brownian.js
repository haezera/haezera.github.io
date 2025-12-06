function randn() {
  let u = 0, v = 0;

  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();

  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

export function gbm(S0, mu, sigma, T, N) {
  const dt = T / N;
  const path = new Array(N + 1);
  path[0] = {
    t: 0,
    p: S0
  };

  for (let i = 1; i <= N; i++) {
    const z = randn();

    path[i] = {
      t: i,
      p: path[i - 1].p * Math.exp(
        (mu - 0.5 * sigma ** 2) * dt +
        sigma * Math.sqrt(dt) * z
      )
    }
  }

  return path;
}