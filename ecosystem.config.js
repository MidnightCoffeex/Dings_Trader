module.exports = {
  apps: [
    // --- BACKEND API (Forecast+PPO cutover) ---
    {
      name: "dt-api",
      script: "api.py",
      cwd: "./ml",
      interpreter: "/home/maxim/.openclaw/workspace/projects/dings-trader/TraderHimSelf/venv/bin/python",
      args: "",
      env: {
        PORT: 8000,
        PYTHONUNBUFFERED: "1"
      },
      log_date_format: "YYYY-MM-DD HH:mm:ss"
    },

    // --- FRONTEND ---
    {
      name: "dt-ui",
      script: "npm",
      args: "start -- --hostname 0.0.0.0 --port 3000",
      cwd: "./ui",
      env: {
        NODE_ENV: "production"
      },
      log_date_format: "YYYY-MM-DD HH:mm:ss"
    },

    // --- PAPER LOOP (single active model) ---
    {
      name: "dt-loop-ppo-v1",
      script: "paper_inference_loop.py",
      cwd: "./ml",
      interpreter: "/home/maxim/.openclaw/workspace/projects/dings-trader/TraderHimSelf/venv/bin/python",
      args: "--model-id paper_ppo_v1 --interval 60 --create-account",
      restart_delay: 5000,
      log_date_format: "YYYY-MM-DD HH:mm:ss"
    }
  ]
};
