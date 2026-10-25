import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = fileURLToPath(new URL("../", import.meta.url));
const args = process.argv.slice(2);
let host = "0.0.0.0";
let port = "7100";

for (let index = 0; index < args.length; index += 1) {
  const arg = args[index];
  if (arg === "--port" && args[index + 1]) port = args[++index];
  else if (arg.startsWith("--port=")) port = arg.split("=")[1];
  else if (arg === "--host" && args[index + 1] && !args[index + 1].startsWith("--")) host = args[++index];
  else if (arg.startsWith("--host=")) host = arg.split("=")[1];
}

const localPython = process.platform === "win32"
  ? path.join(root, ".venv", "Scripts", "python.exe")
  : path.join(root, ".venv", "bin", "python");
const python = existsSync(localPython) ? localPython : "python";
const streamlitArgs = [
  "-m", "streamlit", "run", "app.py",
  "--server.address", host,
  "--server.port", String(port),
  "--server.headless", "true",
  "--browser.gatherUsageStats", "false",
];

const child = spawn(python, streamlitArgs, {
  cwd: root,
  stdio: "inherit",
  env: { ...process.env, PYTHONUNBUFFERED: "1" },
});

for (const signal of ["SIGINT", "SIGTERM"]) {
  process.on(signal, () => child.kill(signal));
}
child.on("error", (error) => {
  console.error(`Unable to start Streamlit: ${error.message}`);
  process.exit(1);
});
child.on("exit", (code) => process.exit(code ?? 0));
