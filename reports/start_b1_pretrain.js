const fs = require("fs");
const path = require("path");
const { spawn } = require("child_process");

const cwd = "E:\\MicroLM";
const stdoutPath = path.join(cwd, "reports", "b1_pretrain_stdout.log");
const stderrPath = path.join(cwd, "reports", "b1_pretrain_stderr.log");
const pidPath = path.join(cwd, "reports", "b1_pretrain.pid");
const statusPath = path.join(cwd, "reports", "b1_pretrain_status.txt");
const exitPath = path.join(cwd, "reports", "b1_pretrain.exitcode");

try {
  fs.rmSync(exitPath, { force: true });
} catch {}

fs.writeFileSync(statusPath, `Started at ${new Date().toISOString()}\n`);

const stdout = fs.openSync(stdoutPath, "w");
const stderr = fs.openSync(stderrPath, "w");
const child = spawn(
  path.join(cwd, ".venv", "Scripts", "python.exe"),
  [
    "scripts\\train_pretrain.py",
    "--config",
    "configs\\pretrain_full_corpus.json",
    "--wandb_mode",
    "disabled",
  ],
  {
    cwd,
    detached: true,
    stdio: ["ignore", stdout, stderr],
    windowsHide: true,
  },
);

fs.writeFileSync(pidPath, `${child.pid}\n`);
fs.appendFileSync(statusPath, `PID ${child.pid}\n`);
child.unref();
console.log(`Started B1 pretrain PID=${child.pid}`);
console.log(`Stdout=${stdoutPath}`);
console.log(`Stderr=${stderrPath}`);
